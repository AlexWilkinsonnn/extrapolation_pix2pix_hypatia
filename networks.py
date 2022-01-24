import sys

import functools
import torch
from torch.nn import init
import torch.nn as nn
from torch.cuda.amp import custom_bwd, custom_fwd


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.n_epochs) / float(opt.n_epochs_decay + 1)

            return lr_l

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif opt.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)

    elif opt.lr_policy == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)

    elif opt.lr_policy == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)

    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)

    return scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)

    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)

    elif norm_type == 'none':
        def norm_layer(x): return Identity()

    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__

        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)

            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)

            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)

            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], no_DataParallel=False): # Need no_DataParallel = True to trace.
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if not no_DataParallel:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs

    init_weights(net, init_type, init_gain=init_gain)

    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal',
    init_gain=0.02, gpu_ids=[], output_layer='tanh', kernel_size=4, outer_stride=2, inner_stride_1=2):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for multiples of 128x128 input images) and 
        [unet_256] (for multiples of 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).

    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, output_layer=output_layer)

    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, output_layer=output_layer)
        
    elif netG == 'unet_128':
        net = UnetGenerator(
            input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, output_layer=output_layer, kernel_size=kernel_size, outer_stride=outer_stride, inner_stride_1=inner_stride_1)

    elif netG == 'unet_256':
        net = UnetGenerator(
            input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, output_layer=output_layer, kernel_size=kernel_size, outer_stride=outer_stride, inner_stride_1=inner_stride_1)

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)

    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)

    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)

    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)

    return init_net(net, init_type, init_gain, gpu_ids)


# from matplotlib import pyplot as plt
# import matplotlib
# import time
# import numpy as np

def CustomLossHitBiasEstimator(input_, output, target, mask, mask_type):
    if mask_type == 'saved':
        if mask.sum() == 0: # NEED OR ELSE OUTPUTS COLLAPSE TO NAN FOR WHATEVER REASON
            return 0, 0, 0, 0, 0, 0
        
        loss_abs_pix = ((mask * output).abs() - (mask * target).abs()).sum()/mask.sum()

        denom = (mask * target).abs()
        denom[denom == 0] = 1 # prevent division by zero
        loss_abs_pix_fractional = (((mask * output).abs() - (mask * target).abs())/denom).sum()/mask.sum()

        loss_channel_positive = ((mask * output * (target >= 0)).sum(3) - (mask * target * (target >= 0)).sum(3)).sum()/mask.sum(3).count_nonzero()
        loss_channel_negative = ((mask * output * (target < 0)).sum(3) - (mask * target * (target < 0)).sum(3)).sum()/mask.sum(3).count_nonzero()
        loss_channel = (loss_channel_negative, loss_channel_positive)

        denom = (mask * target * (target >= 0)).sum(3)
        denom[denom == 0] = 1
        loss_channel_positive_fractional = (((mask * output * (target >= 0)).sum(3) - (mask * target * (target >= 0)).sum(3))/denom).sum()/mask.sum(3).count_nonzero()
        denom = (mask * target * (target < 0)).sum(3)
        denom[denom == 0] = 1
        loss_channel_negative_fractional = (((mask * output * (target < 0)).sum(3) - (mask * target * (target < 0)).sum(3))/denom).sum()/mask.sum(3).count_nonzero()
        loss_channel_fractional = (loss_channel_negative_fractional, loss_channel_positive_fractional)

        loss_event_positive = ((mask * output * (target >= 0)).sum() - (mask * target * (target >= 0)).sum())
        loss_event_negative = ((mask * output * (target < 0)).sum() - (mask * target * (target < 0)).sum())
        loss_event = (loss_event_negative, loss_event_positive)

        loss_event_positive_fractional = loss_event_negative/(mask * target * (target >= 0)).sum()
        loss_event_negative_fractional = loss_event_positive/(mask * target * (target < 0)).sum()
        loss_event_fractional = (loss_event_negative_fractional, loss_event_positive_fractional)
        # print(loss_abs_pix)
        # print(loss_abs_pix_fractional)
        # print(loss_channel)
        # print(loss_channel_fractional)
        # print(loss_event)
        # print(loss_event_fractional)

    elif mask_type == 'none':
        raise NotImplementedError()

    return loss_abs_pix, loss_abs_pix_fractional,  loss_channel, loss_channel_fractional, loss_event, loss_event_fractional

def CustomLoss(input_, output, target, direction, mask, B_ch0_scalefactor, mask_type, nonzero_L1weight, rms):
    """
    """
    if direction == 'AtoB': # SimEnergyDeposit to RawDigit
        if mask_type == 'auto':
            # active_mask = input_[0][0].abs().sum(1).bool()
            # input_chs, output_chs, target_chs = input_[0][0][active_mask, :], output[0][0][active_mask, :], target[0][0][active_mask, :]
            input_chs, output_chs, target_chs = input_[0][0][:, :], output[0][0][:, :], target[0][0][:, :]

            # right_roll10 = target_chs.roll(10, 1)
            # right_roll10[:, :10] = 0
            # left_roll10 = target_chs.roll(-10, 1)
            # left_roll10[:, -10:] = 0
            # right_roll20 = target_chs.roll(20, 1)
            # right_roll20[:, :20] = 0
            # left_roll20 = target_chs.roll(-20, 1)
            # left_roll20[:, -20:] = 0
            right_roll10 = target_chs.roll(5, 1)
            right_roll10[:, :10] = 5*B_ch0_scalefactor
            left_roll10 = target_chs.roll(-5, 1)
            left_roll10[:, -10:] = 5*B_ch0_scalefactor
            # right_roll20 = target_chs.roll(10, 1)
            # right_roll20[:, :20] = 0
            # left_roll20 = target_chs.roll(-10, 1)
            # left_roll20[:, -20:] = 0
            peak_mask = target_chs + right_roll10 + left_roll10# + left_roll20 + right_roll20 # Smearing the adc to find regions with peaks.
            roll1 = target_chs.roll(1, 0)
            roll1[:1] = 5*B_ch0_scalefactor
            roll2 = target_chs.roll(-1, 0)
            roll2[-1:] = 5*B_ch0_scalefactor
            peak_mask = peak_mask + roll1 + roll2
            peak_mask = (peak_mask > (40*B_ch0_scalefactor)) # 40

            if peak_mask.sum() == 0:
                return 0, 0
            
            if (input_chs * peak_mask).sum() == 0:
                return 0, 0

            # ticks = np.arange(1, 513)
            # weights = target_chs[5,:].detach().cpu().numpy()
            # weights[peak_mask[5,:].detach().cpu().numpy()] = np.nan
            # weights_inv = target_chs[5,:].detach().cpu().numpy()
            # weights_inv[~peak_mask[5,:].detach().cpu().numpy()] = np.nan
            # plt.hist(ticks, bins=len(ticks), weights=weights, histtype='step', color='y', label='D', linewidth=3)
            # plt.hist(ticks, bins=len(ticks), weights=weights_inv, histtype='step', color='c', label='D+L1', linewidth=3)
            # plt.ylim(bottom=-30)
            # handle1 = matplotlib.lines.Line2D([], [], c='y')
            # handle2 = matplotlib.lines.Line2D([], [], c='c')
            # plt.legend(handles=[handle1, handle2], labels=['D', 'D+L1'], frameon=False, loc='upper right', fontsize=20)
            # plt.show()

            loss_pix = ((peak_mask * output_chs) - (peak_mask * target_chs)).abs().sum()/peak_mask.sum() # L1
            loss_channel = ((peak_mask * target_chs).sum(1) - (peak_mask * output_chs).sum(1)).abs().sum()/peak_mask.sum(1).count_nonzero()
            
        elif mask_type == 'saved':
            if mask.sum() == 0: # NEED OR ELSE OUTPUTS COLLAPSE TO NAN FOR WHATEVER REASON
                return 0, 0

            # fig, ax = plt.subplots(1, 2)
            # im = target.detach().cpu().numpy()
            # im_masked = (target * mask).detach().cpu().numpy()
            # im_mask = mask.detach().cpu().numpy()
            # ax[0].imshow(im, aspect='auto', interpolation='none', cmap='jet')
            # ax[1].imshow(im_masked, aspect='auto', interpolation='none', cmap='jet')
            # plt.show()

            loss_pix = (((mask * output) - (mask * target)).abs().sum()/mask.sum())/target.size()[0]
            if target.size()[2] == 800: # induction view
                loss_channel_positive = ((mask * target * (target >= 0)).sum(3) - (mask * output * (target >= 0)).sum(3)).abs().sum()/mask.sum(3).count_nonzero()
                loss_channel_negative = ((mask * target * (target < 0)).sum(3) - (mask * output * (target < 0)).sum(3)).abs().sum()/mask.sum(3).count_nonzero()
                loss_channel = ((loss_channel_negative + loss_channel_positive)/2)/target.size()[0]
            else:
                loss_channel = (((mask * target).sum(3) - (mask * output).sum(3)).abs().sum()/mask.sum(3).count_nonzero())/target.size()[0]

            # print("{} {}".format(loss_pix, loss_channel))

        elif mask_type =='saved_1rms':
            if target.size()[2] == 800:
                raise NotImplementedError("induction view not implemented yet")

            if mask.sum() == 0:
                return 0, 0

            loss_pix = ((mask * output) - (mask * target)).abs()
            loss_pix[loss_pix <= (rms*B_ch0_scalefactor)] = 0 
            loss_pix = (loss_pix.sum()/mask.sum())/target.size()[0]

            loss_channel = (((mask * target).sum(3) - (mask * output).sum(3)).abs().sum()/mask.sum(3).count_nonzero())/target.size()[0]
            
        elif mask_type == 'none':
            if target.size()[2] == 800:
                raise NotImplementedError("induction view not implemented yet")
            if target.size()[0] > 1:
                raise NotImplementedError("batch loss not implemented yet")

            loss_pix = (output - target).abs().mean()
            loss_channel = (target.sum(3) - output.sum(3)).abs().sum()/target.size()[2]

        elif mask_type == 'none_weighted':
            if target.size()[2] == 800:
                raise NotImplementedError("induction view not implemented yet")
            if target.size()[0] > 1:
                raise NotImplementedError("batch loss not implemented yet")

            if (target != 0).sum() == 0 or (target == 0).sum() == 0:
                return 0, 0

            loss_pix_nonzero = (output * (target != 0) - target * (target != 0)).abs().sum()/(target != 0).sum()
            loss_pix_zero = (output * (target == 0) - target * (target == 0)).abs().sum()/(target == 0).sum()
            loss_pix = loss_pix_zero + nonzero_L1weight*loss_pix_nonzero
            loss_channel_nonzero = ((target * (target != 0)).sum(3) - (output * (target != 0)).sum(3)).abs().sum()/(target != 0).sum(3).count_nonzero()
            loss_channel_zero = ((target * (target == 0)).sum(3) - (output * (target == 0)).sum(3)).abs().sum()/(target == 0).sum(3).count_nonzero()
            loss_channel = loss_channel_zero + nonzero_L1weight*loss_channel_nonzero

    elif direction == 'BtoA': # RawDigit to SimEnergyDeposit
        loss_pix = (output - target).abs().mean()
        loss_channel = (target.sum(3) - output.sum(3)).abs().sum()/target.size()[2]

    return loss_pix, loss_channel



# def ChannelSumLoss(output, target):
#     """Loss function ensuring correct total ADC on channels.

#     Calculates absolute (need to reconsider this for induction view) differences in channel sums.
#     """
#     loss = (target.sum(3) - output.sum(3)).abs().sum()/target.size()[2]

#     return loss


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()

        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()

        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()

        elif gan_mode in ['wgangp']:
            self.loss = None

        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label

        else:
            target_tensor = self.fake_label

        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and ground truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)

        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()

            else:
                loss = prediction.mean()

        return loss


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, output_layer='tanh', kernel_size=4, outer_stride=2, inner_stride_1=2):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        if kernel_size == 4 and outer_stride == 2 and inner_stride_1 == 2:
            paddings = { 'inner' : 1, 'outer' : 1, 'in1' : 1 }
            output_paddings = { 'inner' : 0, 'outer' : 0, 'in1' : 0 }
        elif kernel_size == (3,5) and outer_stride == 2 and inner_stride_1 == 2:
            paddings = { 'inner' : (1,2), 'outer' : (1,2), 'in1' : (1,2) }
            output_paddings = { 'inner' : 1, 'outer' : 1, 'in1' : 1 }
        elif kernel_size == (3,5) and outer_stride == (1,3) and inner_stride_1 == 2:
            paddings = { 'inner' : (1,2), 'outer' : 1, 'in1' : (1,2) }
            output_paddings = { 'inner' : 1, 'outer' : 0, 'in1' : 1 }
        elif kernel_size == (3,5) and outer_stride == 2 and inner_stride_1 == (1,3):
            paddings = { 'inner' : (1,2), 'outer' : (1,2), 'in1' : 1 }
            output_paddings = { 'inner' : 1, 'outer' : (1,1), 'in1' : 0 }
        elif kernel_size == 3 and outer_stride == (1,3) and inner_stride_1 == (1,3): 
            paddings = { 'inner' : 1, 'outer' : 1, 'in1' : 1 }
            output_paddings = { 'inner' : 1, 'outer' : (0,2), 'in1' : (0,2) }

        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, 
                                             kernel_size=kernel_size, padding=paddings['inner'], output_padding=output_paddings['inner'])  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout,
                                                 kernel_size=kernel_size, padding=paddings['inner'], output_padding=output_paddings['inner'])
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, kernel_size=kernel_size,
                                             padding=paddings['inner'], output_padding=output_paddings['inner'])
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, kernel_size=kernel_size,
                                             padding=paddings['inner'], output_padding=output_paddings['inner'])
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, kernel_size=kernel_size, 
                                             stride=inner_stride_1, padding=paddings['in1'], output_padding=output_paddings['in1']) # , in1=True) # in1=True for legacy, uncomment when needed
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, output_layer=output_layer,
                                             kernel_size=kernel_size, stride=outer_stride, padding=paddings['outer'], output_padding=output_paddings['outer'])  # add the outermost layer

    def forward(self, input):
        return self.model(input)


# For debugging
class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        print(x.size())
        return x

# Custom clamp operation implemetation
class CustomClamp(torch.autograd.Function):
    """
    torch.clamp operatorion in forward pass but gradient is 1 everywhere like an identity.
    I want this to enforce saturation in 12 bit ADC output.
    """
    @staticmethod
    @custom_fwd
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def custom_clamp(input, min, max):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return CustomClamp.apply(input, min, max)


class CustomClampLayer(nn.Module):
    """
    custom_clamp module implementation.
    """
    def __init__(self, min, max):
        super(CustomClampLayer, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return custom_clamp(x, self.min, self.max)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 output_layer='tanh', kernel_size=4, stride=2, padding=0, output_padding=0, in1=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d

        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=kernel_size, stride=stride,
                                        padding=padding, output_padding=output_padding)

            downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias)
            down = [downconv]
            if output_layer == 'tanh':
                up = [uprelu, upconv, nn.Tanh()]

            elif output_layer == 'tanh+clampcollection':
                up = [uprelu, upconv, nn.Tanh(), CustomClampLayer(-0.28169014084507044, 1)] # [-900, 3195]

            elif output_layer == 'tanh+clampinduction':
                up = [uprelu, upconv, nn.Tanh(), CustomClampLayer(-1, 0.7425531914893617)] # [-2350, 1745]

            elif output_layer == 'linear':
                up = [uprelu, upconv, nn.Linear(512, 512)] # will need to change 512 is using different image sizes

            elif output_layer == 'identity':
                up = [uprelu, upconv, nn.Identity()]

            elif output_layer == 'relu':
                up = [uprelu, upconv, nn.ReLU(True)]

            else:
                raise NotImplementedError('output_layer %s not implemented' % output_layer)

            model = down + [submodule] + up

        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=kernel_size, stride=stride,
                                        padding=padding, bias=use_bias, output_padding=output_padding)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up

        # Legacy, uncommment when needed
        # elif in1:
        #     upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)

        #     down = [downrelu, downconv, downnorm]
        #     up = [uprelu, upconv, upnorm]

        #     if use_dropout:
        #         model = down + [submodule] + up + [nn.Dropout(0.5)]

        #     else:
        #         model = down + [submodule] + up

        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias, output_padding=output_padding)

            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]

            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        # print("x.size()={}".format(x.size()))
#        print("self.model(x)={}".format(self.model(x).size()))
        if self.outermost:
            return self.model(x)

        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
        padding_type='reflect', output_layer='tanh'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d

        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if output_layer == 'tanh':
            model += [nn.Tanh()]

        elif output_layer == 'linear':
            model += [nn.Linear(output_nc*512, output_nc*512)]

        elif output_layer == 'identity':
            model += [nn.Identity()]

        elif output_layer == 'relu':
            up = [uprelu, upconv, nn.ReLU(True)]

        else:
            raise NotImplementedError('output_layer %s not implemented' % output_layer)

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]

        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]

        elif padding_type == 'zero':
            p = 1

        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]

        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]

        elif padding_type == 'zero':
            p = 1#

        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)  # add skip connections

        return out


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d

        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d

        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)
