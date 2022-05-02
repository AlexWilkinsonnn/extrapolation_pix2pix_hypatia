import os
from collections import OrderedDict

import torch

import networks


class Pix2pix():
    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.isTrain = opt.isTrain
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0 
        torch.backends.cudnn.benchmark = True

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_pix', 'G_channel', 'D_real', 'D_fake']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain: # options["isTrain"]
            self.model_names = ['G', 'D']

        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.G_output_layer, opt.kernel_size, opt.outer_stride, opt.inner_stride_1)

        if self.isTrain:  
            if opt.unconditional_D:
                self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                    opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            else:
                # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
                self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                    opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            # self.criterionL1 = torch.nn.L1Loss()
            self.criterionCustomLoss = networks.CustomLoss
            # self.criterionChannel = networks.ChannelSumLoss # This is a function
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.adam_weight_decay)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.adam_weight_decay)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        if not self.isTrain:
            load_suffix = opt.epoch #'iter_%d' % opt.epoch
            self.load_networks(load_suffix)

        self.print_networks(False)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def train(self):
        """Make models train mode during train time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def test(self, half=False):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward(half)

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)

            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                # Sometimes I just want to use the model to store the data for other uses so might not have fake_B
                if name == 'fake_B':
                    try:
                        visual_ret[name] = getattr(self, name)
                    except:
                        continue
                else:
                    visual_ret[name] = getattr(self, name)

        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number

        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])

                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))

            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))

        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module

                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))

                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()

                if verbose:
                    print(net)

                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.mask = input['mask'].to(self.device) if self.opt.mask_type.startswith('saved') else False
        self.image_paths = input['A_paths']

        ch_offset = self.opt.channel_offset
        tick_offset = self.opt.tick_offset
        using_offset = True if ch_offset and tick_offset else False # need to have both non-zero offsets if want to use offset
        self.mask = self.mask[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset] if using_offset else self.mask

    # Need these two for making the torchscript.
    def get_real_A(self):
        return self.real_A

    def get_netG(self):
        return self.netG

    def forward(self, hp=False):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if hp:
            self.fake_B = self.netG.half()(self.real_A)
        else:
            self.fake_B = self.netG(self.real_A)  # G(A)
        
    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        ch_offset = self.opt.channel_offset
        tick_offset = self.opt.tick_offset
        using_offset = True if ch_offset and tick_offset else False
        # Fake; stop backprop to the generator by detaching fake_B
        if not self.opt.unconditional_D:
            fake_AB = torch.cat((self.real_A[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset], self.fake_B[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset]), 1) if using_offset else torch.cat((self.real_A, self.fake_B), 1)

            pred_fake = self.netD(fake_AB.detach())

        else:
            pred_fake = self.netD(self.fake_B[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset].detach()) if using_offset else self.netD(self.fake_B.detach())

        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        if not self.opt.unconditional_D:
            real_AB = torch.cat((self.real_A[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset], self.real_B[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset]), 1) if using_offset else torch.cat((self.real_A, self.real_B), 1) # we use conditional GANs; we need to feed both input and output to the discriminator

            pred_real = self.netD(real_AB)

        else:
            pred_real = self.netD(self.real_B[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset]) if using_offset else self.netD(self.real_B)

        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        ch_offset = self.opt.channel_offset
        tick_offset = self.opt.tick_offset
        using_offset = True if ch_offset and tick_offset else False
        # First, G(A) should fake the discriminator
        if not self.opt.unconditional_D:
            fake_AB = torch.cat((self.real_A[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset], self.fake_B[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset]), 1) if using_offset else torch.cat((self.real_A, self.fake_B), 1)

            pred_fake = self.netD(fake_AB)
        
        else:
            pred_fake = self.netD(self.fake_B[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset]) if using_offset else self.netD(self.fake_B)

        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        # self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        if using_offset:
            self.loss_G_pix, self.loss_G_channel = self.criterionCustomLoss(self.real_A[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset], self.fake_B[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset], self.real_B[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset], self.opt.direction, self.mask, self.opt.B_ch0_scalefactor, self.opt.mask_type, self.opt.nonzero_L1weight, self.opt.rms)
        else:
            self.loss_G_pix, self.loss_G_channel = self.criterionCustomLoss(self.real_A, self.fake_B, self.real_B, self.opt.direction, self.mask, self.opt.B_ch0_scalefactor, self.opt.mask_type, self.opt.nonzero_L1weight, self.opt.rms)

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.opt.lambda_pix * self.loss_G_pix + self.opt.lambda_channel * self.loss_G_channel
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
