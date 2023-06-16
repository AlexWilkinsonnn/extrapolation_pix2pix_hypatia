import functools
import torch
import torch.nn as nn
from torch.cuda.amp import custom_bwd, custom_fwd


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer -- the optimizer of the network
        opt -- stores all the experiment flags.
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
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1
        )

    elif opt.lr_policy == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5
        )

    elif opt.lr_policy == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.n_epochs, eta_min=0
        )

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
    For InstanceNorm, we do not use learnable affine parameters.
    We do not track running statistics.
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
        net (network) -- network to be initialized
        init_type (str) -- the name of an initialization method:
            normal | xavier | kaiming | orthogonal
        init_gain (float) -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__

        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)

            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)

            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)

            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type
                )

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_id=0):
    """Initialize a network: 1. register CPU/GPU device; 2. initialize the network weights
    Parameters:
        net (network) -- the network to be initialized
        init_type (str) -- the name of an initialization method:
            normal | xavier | kaiming | orthogonal
        gain (float) -- scaling factor for normal, xavier and orthogonal.
        gpu_id (int) -- which GPU the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    assert(torch.cuda.is_available())
    net.to(gpu_id)

    init_weights(net, init_type, init_gain=init_gain)

    return net


def define_G(
    input_nc, output_nc, ngf, netG,
    norm='batch',
    use_dropout=False,
    init_type='normal',
    init_gain=0.02,
    gpu_id=0,
    output_layer='tanh',
    padding_type='reflect'
):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name:
            resnet_9blocks | resnet_9blocks_downres(4,10)_{1,2} |
            resnet_9blocks_downres(8,8)_{1,2,3} | resnet_6blocks |
            unet_256 | unet_128 | unet_256_k3 | unet_256_k3-5 | unet_256_k3-5_strides1 |
            unet_256_k3-5_strides2 | unet_256_k3_strides1
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str) -- the name of our initialization method.
        init_gain (float) -- scaling factor for normal, xavier and orthogonal.
        gpu_id (int) -- which GPU the network runs on: e.g., 0,1,2
        output_layer (str) -- output layer of G
        padding_type (str) -- type of padding to be used in convolutions
            (only implemented for resnet)

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for multiples of 128x128 input images) and
        [unet_256] (for multiples of 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and
        [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few
        downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project
        (https://github.com/jcjohnson/fast-neural-style).

    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(
            input_nc, output_nc, ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            n_blocks=9,
            output_layer=output_layer,
            padding_type=padding_type,
            n_downsampling=2,
            downsampling_strides=[2, 2],
            upsampling_strides=[2, 2],
            downupsampling_more_features=[True, True],
            upsampling_output_padding=[1, 1]
        )

    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(
            input_nc, output_nc, ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            n_blocks=6,
            output_layer=output_layer,
            padding_type=padding_type,
            n_downsampling=2,
            downsampling_strides=[2, 2],
            upsampling_strides=[2, 2],
            downupsampling_more_features=[True, True],
            upsampling_output_padding=[1, 1]
        )

    elif netG == 'resnet_9blocks_downres(4,10)_1':
        net = ResnetGenerator(
            input_nc, output_nc, ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            n_blocks=9,
            output_layer=output_layer,
            padding_type=padding_type,
            n_downsampling=2,
            downsampling_strides=[2, (2,5)],
            upsampling_strides=[1, 1],
            downupsampling_more_features=[True, True],
            upsampling_output_padding=[0, 0]
        )

    elif netG == 'resnet_9blocks_downres(4,10)_2':
        net = ResnetGenerator(
            input_nc, output_nc, ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            n_blocks=9,
            output_layer=output_layer,
            padding_type=padding_type,
            n_downsampling=3,
            downsampling_strides=[2, 1, (2,5)],
            upsampling_strides=[1, 1, 1],
            downupsampling_more_features=[True, True, True],
            upsampling_output_padding=[0, 0, 0]
        )

    elif netG == 'resnet_9blocks_downres(8,8)_1':
        net = ResnetGenerator(
            input_nc, output_nc, ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            n_blocks=9,
            output_layer=output_layer,
            padding_type=padding_type,
            n_downsampling=3,
            downsampling_strides=[2, 2, 2],
            upsampling_strides=[1, 1, 1],
            downupsampling_more_features=[True, True, True],
            upsampling_output_padding=[0, 0, 0]
        )

    elif netG == 'resnet_9blocks_downres(8,8)_2':
        net = ResnetGenerator(
            input_nc, output_nc, ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            n_blocks=9,
            output_layer=output_layer,
            padding_type=padding_type,
            n_downsampling=5,
            downsampling_strides=[2, 2, 2, 2, 2], # out to in
            upsampling_strides=[1, 1 ,1 ,2 ,2], # out to in
            downupsampling_more_features=[True, True, True, True, True], # out to in
            upsampling_output_padding=[0, 0, 0, 1, 1] # out to in
        )

    elif netG == 'resnet_9blocks_downres(8,8)_3':
        net = ResnetGenerator(
            input_nc, output_nc, ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            n_blocks=9,
            output_layer=output_layer,
            padding_type=padding_type,
            n_downsampling=5,
            downsampling_strides=[2, 2, 2, 2, 2],
            upsampling_strides=[2, 2, 1, 1, 1],
            downupsampling_more_features=[True, False, True, False, True],
            upsampling_output_padding=[1, 1, 1, 1, 1]
        )

    elif netG == 'unet_128':
        net = UnetGenerator(
            input_nc, output_nc, 7, ngf,
            norm_layer=norm_layer, use_dropout=use_dropout, output_layer=output_layer
        )

    elif netG == 'unet_256':
        net = UnetGenerator(
            input_nc, output_nc, 8, ngf,
            norm_layer=norm_layer, use_dropout=use_dropout, output_layer=output_layer
        )

    # if kernel_size == 3 and outer_stride == 2 and inner_stride_1 == 2:
    #     paddings = { 'inner' : 1, 'outer' : 1, 'in1' : 1 }
    #     output_paddings = { 'inner' : 1, 'outer' : 1, 'in1' : 1 }
    elif netG == "unet_256_k3":
        output_paddings = { 0 : 1, 1 : 1, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 1 }
        net = UnetGenerator(
            input_nc, output_nc, 8, ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            output_layer=output_layer,
            kernel_size=3,
            output_paddings=output_paddings
        )

    # elif kernel_size == (3,5) and outer_stride == 2 and inner_stride_1 == 2:
    #     paddings = { 'inner' : (1,2), 'outer' : (1,2), 'in1' : (1,2) }
    #     output_paddings = { 'inner' : 1, 'outer' : 1, 'in1' : 1 }
    elif netG == "unet_256_k3-5":
        paddings = { 0 : (1,2), 1 : (1,2), 2 : 1, 3 : 1, 4 : 1, 5 : 1, 6 : 1, 7 : (1,2) }
        output_paddings = { 0 : 1, 1 : 1, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 1 }
        net = UnetGenerator(
            input_nc, output_nc, 8, ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            output_layer=output_layer,
            kernel_size=(3,5),
            output_paddings=output_paddings,
            paddings=paddings
        )

    # elif kernel_size == (3,5) and outer_stride == (1,3) and inner_stride_1 == 2:
    #     paddings = { 'inner' : (1,2), 'outer' : 1, 'in1' : (1,2) }
    #     output_paddings = { 'inner' : 1, 'outer' : 0, 'in1' : 1 }
    elif netG == "unet_256_k3-5_strides1":
        paddings = { 0 : 1, 1 : (1,2), 2 : 1, 3 : 1, 4 : 1, 5 : 1, 6 : 1, 7 : (1,2) }
        output_paddings = { 0 : 0, 1 : 1, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 1 }
        strides = { 0 : (1,3), 1 : 2,, 2 : 2, 3 : 2, 4 : 2, 5 : 2, 6 : 2, 7 : 2 }
        net = UnetGenerator(
            input_nc, output_nc, 8, ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            output_layer=output_layer,
            kernel_size=(3,5),
            output_paddings=output_paddings,
            paddings=paddings,
            strides=strides
        )

    # elif kernel_size == (3,5) and outer_stride == 2 and inner_stride_1 == (1,3):
    #     paddings = { 'inner' : (1,2), 'outer' : (1,2), 'in1' : 1 }
    #     output_paddings = { 'inner' : 1, 'outer' : (1,1), 'in1' : 0 }
    elif netG == "unet_256_k3-5_strides2":
        paddings = { 0 : (1,2), 1 : 1, 2 : 1, 3 : 1, 4 : 1, 5 : 1, 6 : 1, 7 : (1,2) }
        output_paddings = { 0 : 1, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 1 }
        strides = { 0 : 2, 1 : (1,3), 2 : 2, 3 : 2, 4 : 2, 5 : 2, 6 : 2, 7 : 2 }
        net = UnetGenerator(
            input_nc, output_nc, 8, ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            output_layer=output_layer,
            kernel_size=(3,5),
            output_paddings=output_paddings,
            paddings=paddings,
            strides=strides
        )

    # elif kernel_size == 3 and outer_stride == (1,3) and inner_stride_1 == (1,3):
    #     paddings = { 'inner' : 1, 'outer' : 1, 'in1' : 1 }
    #     output_paddings = { 'inner' : 1, 'outer' : (0,2), 'in1' : (0,2) }
    elif netG == "unet_256_k3_strides1":
        output_paddings = { 0 : (0,2), 1 : (0,2), 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 1 }
        strides = { 0 : (1,3), 1 : (1,3), 2 : 2, 3 : 2, 4 : 2, 5 : 2, 6 : 2, 7 : 2 }
        net = UnetGenerator(
            input_nc, output_nc, 8, ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            output_layer=output_layer,
            kernel_size=3,
            output_paddings=output_paddings,
            paddings=paddings,
            strides=strides
        )

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    return init_net(net, init_type, init_gain, gpu_id)


def define_D(
    input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_id=0
):
    """Create a discriminator

    Parameters:
        input_nc (int) -- the number of channels in input images
        ndf (int) -- the number of filters in the first conv layer
        netD (str) -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int) -- the number of conv layers in the discriminator;
            effective when netD=='n_layers'
        norm (str) -- the type of normalization layers used in the network.
        init_type (str) -- the name of the initialization method.
        init_gain (float) -- scaling factor for normal, xavier and orthogonal.
        gpu_id (int) -- which GPU the network runs on: e.g., 0,1,2

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

    if netD == 'basic': # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)

    elif netD == 'n_layers': # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)

    elif netD == 'pixel': # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)

    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)

    return init_net(net, init_type, init_gain, gpu_id)


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(
        self, input_nc, output_nc, num_downs,
        ngf=64, norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        output_layer='tanh',
        kernel_size=4,
        paddings={},
        output_paddings={},
        strides={}):
        """Construct a Unet generator

        Parameters:
            input_nc (int) -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, if
                |num_downs| == 7, image of size 128x128 will become of size 1x1 at the bottleneck
            ngf (int) -- the number of filters in the last conv layer
            norm_layer -- normalization layer
            output_layer (str) -- output layer to use:
                tanh | tanh+clampinduction | tanh+clampcollection | identity | relu
            kernel_size (int or tuple of int) -- kernel size to use in every layers
            paddings (dict of (int, int or tuple of int)) -- paddings to use at each depth level
            output_paddings (dict of (int, int or tuple of int)) -- output paddings to use at each
                depth level's upsampling
            strides (dict of (int, int or tuple of int)) -- strides to use at each depth level

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()

        if not paddings:
            paddings = { i_layer : 1 for i_layer in range(num_downs) }
        if not output_paddings: # for the upsampling transpose conv only
            output_paddings = { i_layer : 0 for i_layer in range(num_downs) }
        if not strides:
            strides = { i_layer : 2 for i_layer in range(num_downs) }

        assert(num_downs == len(paddings))
        assert(num_downs == len(output_paddings))
        assert(num_downs == len(strides))

        # construct unet structure
        # if kernel_size == 4 and outer_stride == 2 and inner_stride_1 == 2:
        #     paddings = { 'inner' : 1, 'outer' : 1, 'in1' : 1 }
        #     output_paddings = { 'inner' : 0, 'outer' : 0, 'in1' : 0 }
        # if kernel_size == 3 and outer_stride == 2 and inner_stride_1 == 2:
        #     paddings = { 'inner' : 1, 'outer' : 1, 'in1' : 1 }
        #     output_paddings = { 'inner' : 1, 'outer' : 1, 'in1' : 1 }
        # elif kernel_size == (3,5) and outer_stride == 2 and inner_stride_1 == 2:
        #     paddings = { 'inner' : (1,2), 'outer' : (1,2), 'in1' : (1,2) }
        #     output_paddings = { 'inner' : 1, 'outer' : 1, 'in1' : 1 }
        # elif kernel_size == (3,5) and outer_stride == (1,3) and inner_stride_1 == 2:
        #     paddings = { 'inner' : (1,2), 'outer' : 1, 'in1' : (1,2) }
        #     output_paddings = { 'inner' : 1, 'outer' : 0, 'in1' : 1 }
        # elif kernel_size == (3,5) and outer_stride == 2 and inner_stride_1 == (1,3):
        #     paddings = { 'inner' : (1,2), 'outer' : (1,2), 'in1' : 1 }
        #     output_paddings = { 'inner' : 1, 'outer' : (1,1), 'in1' : 0 }
        # elif kernel_size == 3 and outer_stride == (1,3) and inner_stride_1 == (1,3):
        #     paddings = { 'inner' : 1, 'outer' : 1, 'in1' : 1 }
        #     output_paddings = { 'inner' : 1, 'outer' : (0,2), 'in1' : (0,2) }

        itr_layer = reversed(range(num_downs))

        i_layer = next(itr_layer)
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
            kernel_size=kernel_size,
            padding=paddings[i_layer],
            output_padding=output_paddings[i_layer],
            stride=strides[i_layer]
        )  # add the innermost layer

        for _ in range(num_downs - 5): # add intermediate layers with ngf * 8 filters
            i_layer = next(itr_layer)
            unet_block = UnetSkipConnectionBlock(
                ngf * 8, ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
                kernel_size=kernel_size,
                padding=paddings[i_layer],
                output_padding=output_paddings[i_layer],
                stride=strides[i_layer]
            )

        # gradually reduce the number of filters from ngf * 8 to ngf
        i_layer = next(itr_layer)
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            kernel_size=kernel_size,
            padding=paddings[i_layer],
            output_padding=output_paddings[i_layer],
            stride=strides[i_layer]
        )
        i_layer = next(itr_layer)
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            kernel_size=kernel_size,
            padding=paddings[i_layer],
            output_padding=output_paddings[i_layer],
            stride=strides[i_layer]
        )
        i_layer = next(itr_layer)
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            kernel_size=kernel_size,
            padding=paddings[i_layer],
            output_padding=output_paddings[i_layer],
            stride=strides[i_layer]
        )
        i_layer = next(itr_layer)
        self.model = unet_block = UnetSkipConnectionBlock(
            output_nc, ngf,
            input_nc=input_nc,
            submodule=unet_block,
            norm_layer=norm_layer,
            output_layer=output_layer,
            outermost=True,
            use_dropout=use_dropout,
            kernel_size=kernel_size,
            padding=paddings[i_layer],
            output_padding=output_paddings[i_layer],
            stride=strides[i_layer]
        )

    def forward(self, input):
        return self.model(input)


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
    def __init__(
        self, outer_nc, inner_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        output_layer='tanh',
        kernel_size=4,
        padding=0,
        output_padding=0,
        stride=2
    ):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool) -- if this module is the outermost module
            innermost (bool) -- if this module is the innermost module
            norm_layer -- normalization layer
            use_dropout (bool) -- if use dropout layers
            output_layer (str) -- output layer to use:
                tanh | tanh+clampinduction | tanh+clampcollection | identity | relu
            kernel_size (int or tuple of int) -- kernel size to use
            padding (int or tuple of int) -- padding to use
            output_padding (int or tuple of int) -- output padding to use
            stride (int or tuple of int) -- stride to use

        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(
            input_nc, inner_nc,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias
        )
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding
            )

            down = [downconv]
            up = [uprelu, upconv]

            if output_layer == 'tanh':
                up += [nn.Tanh()]
            elif output_layer == 'tanh+clampcollection':
                # [-900, 3195] adc range
                up += [nn.Tanh(), CustomClampLayer(-0.28169014084507044, 1)]
            elif output_layer == 'tanh+clampinduction':
                 # [-2350, 1745] adc range
                up += [nn.Tanh(), CustomClampLayer(-1, 0.7425531914893617)]
            elif output_layer == 'identity':
                up += [nn.Identity()]
            elif output_layer == 'relu':
                up += [nn.ReLU(True)]
            else:
                raise NotImplementedError('output_layer %s not implemented' % output_layer)

            model = down + [submodule] + up

        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc, outer_nc,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=use_bias,
                output_padding=output_padding
            )

            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up

        else:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=use_bias,
                output_padding=output_padding
            )

            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else: # add skip connections
            return torch.cat([x, self.model(x)], 1)


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling
    operations. We adapt Torch code and idea from Justin Johnson's neural style transfer
    project(https://github.com/jcjohnson/fast-neural-style)
    """
    def __init__(
            self, input_nc, output_nc,
            ngf=64,
            norm_layer=nn.BatchNorm2d,
            use_dropout=False,
            n_blocks=6,
            padding_type='reflect',
            output_layer='tanh',
            n_downsampling=2,
            downsampling_strides=[2,2],
            upsampling_strides=[2,2],
            downupsampling_more_features=[True,True],
            upsampling_output_padding=[1,1]
        ):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int) -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            ngf (int) -- the number of filters in the last conv layer
            norm_layer -- normalization layer
            use_dropout (bool) -- if use dropout layers
            n_blocks (int) -- the number of ResNet blocks
            padding_type (str) -- the name of padding layer in conv layers: reflect | zero
            output_layer (str) -- output layer to use:
                tanh | tanh+clampinduction | tanh+clampcollection | identity | relu
            n_downsampling (int) -- number of downsampling/upsampling layers before resnet blocks
            downsampling_strides (list of int or list of tuple of int) -- strides to use for each
                level of downsampling
            upsampling_strides (list of int or list of tuple of int) -- strides to use for each
                level of upsampling
            downupsampling_more_features (list of bool) -- whether to double the number of features
                at each level of downsampling
            upsampling_output_padding (list of int or list of tuple of int) -- output padding for
                each level of upsamling
        """
        assert(n_blocks >= 0)
        assert(n_downsampling == len(downsampling_strides))
        assert(n_downsampling == len(downsampling_more_features))
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if padding_type == 'reflect':
            model = [nn.ReflectionPad2d(3),
                     nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                     norm_layer(ngf),
                     nn.ReLU(True)]
        elif padding_type == 'zeros':
            model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3, bias=use_bias),
                     norm_layer(ngf),
                     nn.ReLU(True)]
        else:
            raise NotImplementedError('padding_type %s not implemented' % padding_type)

        # # add downsampling layers
        # if downres == 'none' or downres == '(4,10)_1':
        #     n_downsampling = 2
        #     strides = [2, (2,5) if downres == '(4,10)_1' else 2]
        #     for i in range(n_downsampling):
        #         mult = 2 ** i
        #         stride = strides[i]

        #         model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=stride, padding=1, bias=use_bias),
        #                   norm_layer(ngf * mult * 2),
        #                   nn.ReLU(True)]

        # elif downres == '(4,10)_2':
        #     n_downsampling = 3
        #     strides = [2, 1, (2,5)]
        #     for i in range(n_downsampling):
        #         mult = 2 ** i
        #         stride = strides[i]

        #         model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=stride, padding=1, bias=use_bias),
        #                   norm_layer(ngf * mult * 2),
        #                   nn.ReLU(True)]

        # elif downres == '(8,8)_1':
        #     n_downsampling = 3
        #     strides = [2, 2, 2]
        #     for i in range(n_downsampling):
        #         mult = 2 ** i
        #         stride = strides[i]

        #         model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=stride, padding=1, bias=use_bias),
        #                   norm_layer(ngf * mult * 2),
        #                   nn.ReLU(True)]

        # elif downres == '(8,8)_2':
        #     n_downsampling = 5
        #     strides = [2, 2, 2, 2, 2]
        #     for i in range(n_downsampling):
        #         mult = 2 ** i
        #         stride = strides[i]

        #         model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=stride, padding=1, bias=use_bias),
        #                   norm_layer(ngf * mult * 2),
        #                   nn.ReLU(True)]

        # elif downres == '(8,8)_3':
        #     n_downsampling = 5
        #     strides = [2, 2, 2, 2, 2]
        #     more_features = [True, False, True, False, True]
        #     mult = 1
        #     for i in range(n_downsampling):
        #         stride = strides[i]

        #         if more_features[i]:
        #             model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=stride, padding=1, bias=use_bias),
        #                       norm_layer(ngf * mult * 2),
        #                       nn.ReLU(True)]
        #             mult *= 2

        #         else:
        #             model += [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, stride=stride, padding=1, bias=use_bias),
        #                       norm_layer(ngf * mult),
        #                       nn.ReLU(True)]

        # else:
        #     raise NotImplementedError("downres type {} not implemented".format(downres))

        # if downres != '(8,8)_3':
        #     mult = 2 ** n_downsampling

        # for i in range(n_blocks):       # add ResNet blocks
        #     model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        # if downres == 'none':
        #     for i in range(n_downsampling):  # add upsampling layers
        #         mult = 2 ** (n_downsampling - i)
        #         model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
        #                                      kernel_size=3, stride=2,
        #                                      padding=1, output_padding=1,
        #                                      bias=use_bias),
        #                   norm_layer(int(ngf * mult / 2)),
        #                   nn.ReLU(True)]

        # elif downres == '(8,8)_2':
        #     for i in range(n_downsampling):  # add some upsampling layers and collapse the remaining feature dimension
        #         mult = 2 ** (n_downsampling - i)
        #         if i < 2:
        #             model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
        #                                          kernel_size=3, stride=2,
        #                                          padding=1, output_padding=1,
        #                                          bias=use_bias),
        #                       norm_layer(int(ngf * mult / 2)),
        #                       nn.ReLU(True)]
        #         else:
        #             model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
        #                                          kernel_size=3, stride=1,
        #                                          padding=1, output_padding=0,
        #                                          bias=use_bias),
        #                       norm_layer(int(ngf * mult / 2)),
        #                       nn.ReLU(True)]

        # elif downres == '(8,8)_3':
        #     more_features_reversed = list(reversed([True, False, True, False, True]))

        #     for i in range(n_downsampling):  # add some upsampling layers and collapse the remaining feature dimension
        #         if i < 2:
        #             if more_features_reversed[i]:
        #                 model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
        #                                              kernel_size=3, stride=2,
        #                                              padding=1, output_padding=1,
        #                                              bias=use_bias),
        #                           norm_layer(int(ngf * mult / 2)),
        #                           nn.ReLU(True)]
        #                 mult = int(mult / 2)

        #             else:
        #                 model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult),
        #                                              kernel_size=3, stride=2,
        #                                              padding=1, output_padding=1,
        #                                              bias=use_bias),
        #                           norm_layer(int(ngf * mult)),
        #                           nn.ReLU(True)]


        #         else:
        #             if more_features_reversed[i]:
        #                 model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
        #                                              kernel_size=3, stride=1,
        #                                              padding=1, output_padding=0,
        #                                              bias=use_bias),
        #                           norm_layer(int(ngf * mult / 2)),
        #                           nn.ReLU(True)]
        #                 mult = int(mult / 2)

        #             else:
        #                 model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult),
        #                                              kernel_size=3, stride=1,
        #                                              padding=1, output_padding=0,
        #                                              bias=use_bias),
        #                           norm_layer(int(ngf * mult)),
        #                           nn.ReLU(True)]

        # else:
        #     for i in range(n_downsampling):  # no upsampling, just collapse feature dimension
        #         mult = 2 ** (n_downsampling - i)
        #         model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
        #                                      kernel_size=3, stride=1,
        #                                      padding=1, output_padding=0,
        #                                      bias=use_bias),
        #                   norm_layer(int(ngf * mult / 2)),
        #                   nn.ReLU(True)]

        # if padding_type == 'reflect':
        #     model += [nn.ReflectionPad2d(3)]
        #     model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        # elif padding_type == 'zeros':
        #     model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
        # else:
        #     raise NotImplementedError('padding_type %s not implemented' % padding_type)

        # add downsampling layers
        mult = 1
        for i in range(n_downsampling):
            in_features = ngf * mult
            if not downupsampling_more_features[i]:
                out_features = ngf * mult
            else:
                out_features = ngf * mult * 2
                mult *= 2

            model += [
                nn.Conv2d(
                    in_features, out_features,
                    kernel_size=3, stride=downsampling_strides[i], padding=1, bias=use_bias
                ),
                norm_layer(out_features),
                nn.ReLU(True)
            ]

        # add ResNet blocks
        for i in range(n_blocks):
            model += [
                ResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias
                )
            ]

        # add upsampling layers
        for i in reversed(range(n_downsampling)):
            in_features = ngf * mult
            if not downupsampling_more_features[i]:
                out_features = ngf * mult
            else:
                out_features = int(ngf * mult / 2)
                mult = int(mult / 2)

            model += [
                nn.ConvTranspose2d(
                    in_features, out_features,
                    kernel_size=3,
                    stride=upsampling_strides[i],
                    output_padding=upsampling_output_padding[i],
                    padding=1,
                    bias=use_bias
                ),
                norm_layer(out_features),
                nn.ReLU(True)
            ]

        if padding_type == 'reflect':
            model += [nn.ReflectionPad2d(3)]
            model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        elif padding_type == 'zeros':
            model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
        else:
            raise NotImplementedError('padding_type %s not implemented' % padding_type)

        if output_layer == 'tanh':
            up += [nn.Tanh()]
        elif output_layer == 'tanh+clampcollection':
            # [-900, 3195] adc range
            up += [nn.Tanh(), CustomClampLayer(-0.28169014084507044, 1)]
        elif output_layer == 'tanh+clampinduction':
             # [-2350, 1745] adc range
            up += [nn.Tanh(), CustomClampLayer(-1, 0.7425531914893617)]
        elif output_layer == 'identity':
            up += [nn.Identity()]
        elif output_layer == 'relu':
            up += [nn.ReLU(True)]
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
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias
        )

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int) -- the number of channels in the conv layer.
            padding_type (str) -- the name of padding layer: reflect | replicate | zero
            norm_layer -- normalization layer
            use_dropout (bool) -- if use dropout layers.
            use_bias (bool) -- if the conv layer uses bias or not
        Returns a conv block
        (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zeros':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True)
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zeros':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x) # add skip connections
        return out


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int) -- the number of channels in input images
            ndf (int) -- the number of filters in the last conv layer
            n_layers (int) -- the number of conv layers in the discriminator
            norm_layer -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        # no need to use bias as BatchNorm2d has affine parameters
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw, padw = 4, 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult, nf_mult_prev = 1, 1
        for n in range(1, n_layers): # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev, ndf * nf_mult,
                    kernel_size=kw, stride=2, padding=padw, bias=use_bias
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev, ndf * nf_mult,
                kernel_size=kw, stride=1, padding=padw, bias=use_bias
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # output 1 channel prediction map
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int) -- the number of channels in input images
            ndf (int) -- the number of filters in the last conv layer
            norm_layer -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        # no need to use bias as BatchNorm2d has affine parameters
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)
        ]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)

