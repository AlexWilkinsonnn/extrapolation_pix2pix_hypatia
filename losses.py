import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) -- the type of GAN objective.
                It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) -- label for a real image
            target_fake_label (bool) -- label of a fake image

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


def CustomLoss(input_, output, target, direction, mask, B_ch0_scalefactor, mask_type, nonzero_L1weight, rms, name=''):
    """
    """
    if direction == 'AtoB': # SimEnergyDeposit to RawDigit
        if mask_type == 'auto':
            # active_mask = input_[0][0].abs().sum(1).bool()
            # input_chs, output_chs, target_chs = input_[0][0][active_mask, :], output[0][0][active_mask, :], target[0][0][active_mask, :]
            input_chs, output_chs, target_chs = input_[0][0][:, :], output[0][0][:, :], target[0][0][:, :]

            right_roll5 = target_chs.roll(5, 1)
            right_roll5[:, :5] = 5*B_ch0_scalefactor
            left_roll5 = target_chs.roll(-5, 1)
            left_roll5[:, -5:] = 5*B_ch0_scalefactor
            peak_mask = target_chs + right_roll5 + left_roll5
            right_roll10 = target_chs.roll(10, 1)
            right_roll10[:, :10] = 5*B_ch0_scalefactor
            left_roll10 = target_chs.roll(-10, 1)
            left_roll10[:, -10:] = 5*B_ch0_scalefactor
            peak_mask = peak_mask + right_roll10 + left_roll10

            roll1 = target_chs.roll(1, 0)
            roll1[:1] = 5*B_ch0_scalefactor
            roll2 = target_chs.roll(-1, 0)
            roll2[-1:] = 5*B_ch0_scalefactor
            peak_mask = peak_mask + roll1 + roll2

            peak_mask = (peak_mask > (65*B_ch0_scalefactor))

            if peak_mask.sum() == 0:
                return 0, 0

            if input_chs.size() == peak_mask.size() and (input_chs * peak_mask).sum() == 0:
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

        elif mask_type == 'saved' or mask_type == 'saved_fd':
            if mask.sum() == 0: # NEED OR ELSE OUTPUTS COLLAPSE TO NAN FOR WHATEVER REASON
                return 0, 0

            if target.size()[2] == 800 or target.size()[2] == 480:
                induction = target.size()[2] == 800
            else:
                induction = False if 'Z' in name else True

            # fig, ax = plt.subplots(1, 2)
            # im = target.detach().cpu().numpy()
            # im_masked = (target * mask).detach().cpu().numpy()
            # im_mask = mask.detach().cpu().numpy()
            # ax[0].imshow(im, aspect='auto', interpolation='none', cmap='jet')
            # ax[1].imshow(im_masked, aspect='auto', interpolation='none', cmap='jet')
            # plt.show()

            loss_pix = (((mask * output) - (mask * target)).abs().sum()/mask.sum())
            if induction:
                loss_channel_positive = ((mask * target * (target >= 0)).sum(3) - (mask * output * (target >= 0)).sum(3)).abs().sum()/mask.sum(3).count_nonzero()
                loss_channel_negative = ((mask * target * (target < 0)).sum(3) - (mask * output * (target < 0)).sum(3)).abs().sum()/mask.sum(3).count_nonzero()
                loss_channel = ((loss_channel_negative + loss_channel_positive)/2)
            else:
                loss_channel = (((mask * target).sum(3) - (mask * output).sum(3)).abs().sum()/mask.sum(3).count_nonzero())

            # print("{} {}".format(loss_pix, loss_channel))

        elif mask_type =='saved_1rms':
            if target.size()[2] == 800:
                raise NotImplementedError("induction view not implemented yet")
            if target.size()[0] > 1:
                raise NotImplementedError("batch loss not implemented yet")

            if mask.sum() == 0:
                return 0, 0

            loss_pix = ((mask * output) - (mask * target)).abs()
            loss_pix[loss_pix <= (rms*B_ch0_scalefactor)] = 0
            loss_pix = (loss_pix.sum()/mask.sum())/target.size()[0]

            loss_channel = (((mask * target).sum(3) - (mask * output).sum(3)).abs().sum()/mask.sum(3).count_nonzero())/target.size()[0]

        elif mask_type == 'none' or mask_type == 'dont_use':
            if target.size()[0] > 1:
                raise NotImplementedError("batch loss not implemented yet")

            loss_pix = (output - target).abs().mean()
            if target.size()[2] == 800:
                loss_channel_positive = ((target * (target >= 0)).sum(3) - (output * (target >= 0)).sum(3)).abs().sum()/target.size()[2]
                loss_channel_negative = ((target * (target < 0)).sum(3) - (output * (target < 0)).sum(3)).abs().sum()/target.size()[2]
                loss_channel = (loss_channel_negative + loss_channel_positive)/2
            else:
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
