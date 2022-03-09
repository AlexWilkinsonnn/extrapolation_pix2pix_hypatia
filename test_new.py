import argparse, os, sys
from collections import namedtuple
from operator import itemgetter

import numpy as np 
import torch
import yaml
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from model import *
from dataset import *
from networks import CustomLoss, CustomLossHitBiasEstimator

plt.rc('font', family='serif')

def main(opt):
    out_dir = os.path.join('/home/awilkins/extrapolation_pix2pix/results', opt.name)
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dataset_test = CustomDatasetDataLoader(opt, valid=True, nd_ped=nd_ped).load_data()
    dataset_test_size = len(dataset_test)
    print("Number of test images={}".format(dataset_test_size))

    model = Pix2pix(opt)
    print("model {} was created".format(type(model).__name__))
    model.setup(opt)
    model.eval()

    losses_pix, losses_channel = [], []
    # losses_abs_pix_bias, losses_abs_pix_bias_fractional = [], [] 
    # losses_channel_bias, losses_channel_bias_fractional = [], []
    # losses_event_bias, losses_event_bias_fractional = [], []
    losses_event_over20, losses_event_over20_fractional = [], []
    losses_event_underneg20, losses_event_underneg20_fractional = [], []
    losses_pix_absover20, losses_channel_absover20 = [], []
    file_losses = { }
    if half_precision:
        name = 'output_images_FP16_epoch{}.pdf'.format(opt.epoch)
    else:
        name = 'output_images_epoch{}.pdf'.format(opt.epoch)
    if test_sample:
        name = 'test_sample_' + name
    pdf = PdfPages(os.path.join(out_dir, name))
    if half_precision:
        name_bias = 'output_biashist_FP16_epoch{}.pdf'.format(opt.epoch)
    else:
        name_bias = 'output_biashist_epoch{}.pdf'.format(opt.epoch)
    if test_sample:
        name_bias = 'test_sample_' + name_bias
    pdf_bias = PdfPages(os.path.join(out_dir, name_bias))
    for i, data in enumerate(dataset_test):
        if half_precision:
            data = { 'A' : data['A'].half(), 'B' : data['B'].half(), 'A_paths' : data['A_paths'],
                'B_paths' : data['B_paths'], 'mask' : data['mask'].half() }

        model.set_input(data)
        model.test(half_precision)

        print(opt.mask_type)
        visuals = model.get_current_visuals()
        ch_offset, tick_offset = opt.channel_offset, opt.tick_offset
        realA = visuals['real_A'].cpu()[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset]
        realB = visuals['real_B'].cpu()[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset]
        fakeB = visuals['fake_B'].cpu()[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset]
        realA/=opt.A_ch0_scalefactor
        realB/=opt.B_ch0_scalefactor
        fakeB/=opt.B_ch0_scalefactor
        mask = data['mask'].cpu()[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset]
        loss_pix, loss_channel = CustomLoss(realA.float(), fakeB.float(), realB.float(), 'AtoB', mask.float(), opt.B_ch0_scalefactor, opt.mask_type, opt.nonzero_L1weight, opt.rms)
        # loss_abs_pix_bias, loss_abs_pix_bias_fractional, loss_channel_bias, loss_channel_bias_fractional, loss_event_bias, loss_event_bias_fractional = CustomLossHitBiasEstimator(
        #     realA.float(), fakeB.float(), realB.float(), mask.float(), opt.mask_type)
        loss_event_over20 = (fakeB.float() * (fakeB.float() > 20)).sum() - (realB.float() * (realB.float() > 20)).sum()
        loss_event_over20_fractional = loss_event_over20/((realB.float() * (realB.float() > 20)).sum())
        loss_event_underneg20 = (fakeB.float() * (fakeB.float() < -20)).sum() - (realB.float() * (realB.float() < -20)).sum()
        loss_event_underneg20_fractional = loss_event_underneg20/((realB.float() * (realB.float() < -20)).sum())
        loss_pix_absover20 = ((fakeB.float() - realB.float()) * ((fakeB.float().abs() > 20) + (realB.float().abs() > 20))).abs().sum()/(((fakeB.float().abs() > 20) + (realB.float().abs() > 20)).sum())
        loss_channel_absover20 = ((fakeB.float() * (fakeB.float().abs() > 20)).sum(3) - (realB.float() * (realB.float().abs() > 20)).sum(3)).abs().sum()/realB.size()[2]
        if loss_pix != 0:
            losses_pix.append(loss_pix.item())
            losses_channel.append(loss_channel.item())
            # losses_abs_pix_bias.append(loss_abs_pix_bias.item())
            # losses_abs_pix_bias_fractional.append(loss_abs_pix_bias_fractional.item())
            # losses_channel_bias.append((loss_channel_bias[0].item(), loss_channel_bias[1].item()))
            # losses_channel_bias_fractional.append((loss_channel_bias_fractional[0].item(), loss_channel_bias_fractional[1].item()))
            # losses_event_bias.append((loss_event_bias[0].item(), loss_event_bias[1].item()))
            # losses_event_bias_fractional.append((loss_event_bias_fractional[0].item(), loss_event_bias_fractional[1].item()))
            if (realB.float() * (realB.float() > 20)).sum() != 0:
                losses_event_over20.append(loss_event_over20.item())
                losses_event_over20_fractional.append(loss_event_over20_fractional.item())
            if (realB.float() * (realB.float() < -20)).sum() != 0:
                losses_event_underneg20.append(loss_event_underneg20.item())
                losses_event_underneg20_fractional.append(loss_event_underneg20_fractional.item())
            if ((fakeB.float().abs() > 20) + (realB.float().abs() > 20)).sum() != 0:
                losses_pix_absover20.append(loss_pix_absover20.item())
                losses_channel_absover20.append(loss_channel_absover20.item())

            file_losses[data['A_paths'][0]] = loss_pix.item()

        if i < 20:
            realA = realA[0, 0].numpy().astype(int)
            realB = realB[0, 0].numpy().astype(int)
            fakeB = fakeB[0, 0].numpy().astype(int)

            # if i == 14:
            #     np.save('/home/awilkins/extrapolation_pix2pix/sample/realA_Z_example.npy', realA)
            #     np.save('/home/awilkins/extrapolation_pix2pix/sample/realB_Z_example.npy', realB)
            #     np.save('/home/awilkins/extrapolation_pix2pix/sample/fakeB_Z_example.npy', fakeB)
            #     sys.exit()
            
            if include_realA:
                fig, ax = plt.subplots(1, 3, figsize=(24, 8))
            else:
                fig, ax = plt.subplots(1, 2, figsize=(16, 8))

            adc_max = max([realB.max(), fakeB.max()])
            adc_min = min([realB.min(), fakeB.min()])
            if realA.shape[0] == 800:
                vmax = max(adc_max, -adc_min)
                vmin = -vmax
                cmap = 'seismic'
            else:
                vmax = adc_max
                vmin = adc_min
                cmap = 'viridis'
                
            # auto-cropping.
            non_zeros = np.nonzero(realA)
            ch_min = non_zeros[0].min() - 10 if (non_zeros[0].min() - 10) > 0 else 0
            ch_max = non_zeros[0].max() + 11 if (non_zeros[0].max() + 11) < 480 else 480
            tick_min = non_zeros[1].min() - 50 if (non_zeros[1].min() - 50) > 0 else 0
            tick_max = non_zeros[1].max() + 51 if (non_zeros[1].max() + 51) < 4492 else 4492
            realA_cropped = realA[ch_min:ch_max, tick_min:tick_max]
            realB_cropped = realB[ch_min:ch_max, tick_min:tick_max]
            fakeB_cropped = fakeB[ch_min:ch_max, tick_min:tick_max]

            if include_realA:
                ax[0].imshow(np.ma.masked_where(realA_cropped == 0, realA_cropped).T, interpolation='none', aspect='auto', cmap='viridis', origin='lower')
                ax[0].set_title("Input", fontsize=16)
                
                ax[1].imshow(realB_cropped.T, interpolation='none', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
                ax[1].set_title("Truth", fontsize=16)
                
                ax[2].imshow(fakeB_cropped.T, interpolation='none', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
                ax[2].set_title("Output", fontsize=16)
            else:
                ax[0].imshow(realB_cropped.T, interpolation='none', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
                ax[0].set_title("Truth", fontsize=16)
                
                ax[1].imshow(fakeB_cropped.T, interpolation='none', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
                ax[1].set_title("Output", fontsize=16)

            for a in ax: a.set_axis_off()

            fig.tight_layout()
            pdf.savefig(bbox_inches='tight')
            plt.close()

            fig, ax = plt.subplots(figsize=(24,8))

            ch = (0, 0)
            for idx, col in enumerate(realA):
                if np.abs(col).sum() > ch[1]:
                    ch = (idx, np.abs(col).sum())
            ch = ch[0]
            start_tick = np.nonzero(realA[ch, :])[0][0] - 200  if np.nonzero(realA[ch, :])[0][0] > 200 else 0
            end_tick = np.nonzero(realA[ch, :])[0][-1] + 200  if np.nonzero(realA[ch, :])[0][-1] < 4292 else 4292
            ticks = np.arange(start_tick + 1, end_tick + 1)

            ax.hist(ticks, bins=len(ticks), weights=realB[ch,start_tick:end_tick], histtype='step', label='real FD adc', linewidth=0.7, color='r')
            ax.hist(ticks, bins=len(ticks), weights=fakeB[ch,start_tick:end_tick], histtype='step', label='output FD adc', linewidth=0.7, color='b')
            ax.set_ylabel("adc", fontsize=14)
            ax.set_xlabel("tick", fontsize=14)
            ax.set_xlim(start_tick + 1, end_tick + 1)

            ax2 = ax.twinx()
            ax2.hist(ticks, bins=len(ticks), weights=realA[ch,start_tick:end_tick], histtype='step', label='ND ADC', linewidth=0.7, color='g')
            ax2.set_ylabel("ND adc", fontsize=14)

            ax_ylims = ax.axes.get_ylim()
            ax_yratio = ax_ylims[0] / ax_ylims[1]
            ax2_ylims = ax2.axes.get_ylim()
            ax2_yratio = ax2_ylims[0] / ax2_ylims[1]
            if ax_yratio < ax2_yratio:
                ax2.set_ylim(bottom = ax2_ylims[1]*ax_yratio)
            else:
                ax.set_ylim(bottom = ax_ylims[1]*ax2_yratio)

            handles, labels = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles += handles2
            labels += labels2
            new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
            plt.legend(handles=new_handles, labels=labels, prop={'size': 14})

            plt.title("Channel {} in ROP".format(ch), fontsize=16)

            fig.tight_layout()
            pdf.savefig(bbox_inches='tight')
            plt.close()

        if i == 20:
            pdf.close()
            print("Output pdf done")

        if i % 400 == 0:
            print("{}/{}".format(i, dataset_test_size))

    if i < 20:
        pdf.close()

    if test_sample:
        sys.exit()

    print("Making pdf of worst images")
    if half_precision:
        pdf2 = PdfPages(os.path.join(out_dir, 'worst_output_images_FP16_epoch{}.pdf'.format(opt.epoch)))
    else:
        pdf2 = PdfPages(os.path.join(out_dir, 'worst_output_images_epoch{}.pdf'.format(opt.epoch)))    
    worst_files = dict(sorted(file_losses.items(), key=itemgetter(1), reverse=True)[:20])
    for i, data in enumerate(dataset_test):
        if data['A_paths'][0] not in worst_files.keys():
            continue

        if half_precision:
            data = { 'A' : data['A'].half(), 'B' : data['B'].half(), 'A_paths' : data['A_paths'],
                'B_paths' : data['B_paths'], 'mask' : data['mask'].half() }

        model.set_input(data)
        model.test(half_precision)

        visuals = model.get_current_visuals()
        ch_offset, tick_offset = opt.channel_offset, opt.tick_offset
        realA = visuals['real_A'].cpu()[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset]
        realB = visuals['real_B'].cpu()[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset]
        fakeB = visuals['fake_B'].cpu()[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset]
        realA/=opt.A_ch0_scalefactor
        realB/=opt.B_ch0_scalefactor
        fakeB/=opt.B_ch0_scalefactor
        mask = data['mask'].cpu()[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset]
        loss_pix, loss_channel = CustomLoss(realA.float(), fakeB.float(), realB.float(), 'AtoB', mask.float(), opt.B_ch0_scalefactor, opt.mask_type, opt.nonzero_L1weight, opt.rms)

        realA = realA[0, 0].numpy().astype(int)
        realB = realB[0, 0].numpy().astype(int)
        fakeB = fakeB[0, 0].numpy().astype(int)

        if include_realA:
            fig, ax = plt.subplots(1, 3, figsize=(24, 8))
        else:
            fig, ax = plt.subplots(1, 2, figsize=(16, 8))

        adc_max = max([realB.max(), fakeB.max()])
        adc_min = min([realB.min(), fakeB.min()])
        if realA.shape[0] == 800:
            vmax = max(adc_max, -adc_min)
            vmin = -vmax
            cmap = 'seismic'
        else:
            vmax = adc_max
            vmin = adc_min
            cmap = 'viridis'

        if include_realA:
            ax[0].imshow(np.ma.masked_where(realA_cropped == 0, realA_cropped).T, interpolation='none', aspect='auto', cmap='viridis', origin='lower')
            ax[0].set_title("Input", fontsize=16)
                
            ax[1].imshow(realB_cropped.T, interpolation='none', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            ax[1].set_title("Truth", fontsize=16)
                
            ax[2].imshow(fakeB_cropped.T, interpolation='none', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            ax[2].set_title("Output", fontsize=16)
        else:
            ax[0].imshow(realB_cropped.T, interpolation='none', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            ax[0].set_title("Truth", fontsize=16)
            
            ax[1].imshow(fakeB_cropped.T, interpolation='none', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            ax[1].set_title("Output", fontsize=16)

        for a in ax: a.set_axis_off()

        fig.tight_layout()
        pdf2.savefig(bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(24,8))

        ch = (0, 0)
        for idx, col in enumerate(realA):
            if np.abs(col).sum() > ch[1]:
                ch = (idx, np.abs(col).sum())
        ch = ch[0]
        start_tick = np.nonzero(realA[ch, :])[0][0] - 200  if np.nonzero(realA[ch, :])[0][0] > 200 else 0
        end_tick = np.nonzero(realA[ch, :])[0][-1] + 200  if np.nonzero(realA[ch, :])[0][-1] < 4292 else 4292
        ticks = np.arange(start_tick + 1, end_tick + 1)

        ax.hist(ticks, bins=len(ticks), weights=realB[ch,start_tick:end_tick], histtype='step', label='real FD adc', linewidth=0.7, color='r')
        ax.hist(ticks, bins=len(ticks), weights=fakeB[ch,start_tick:end_tick], histtype='step', label='output FD _adc', linewidth=0.7, color='b')
        ax.set_ylabel("FD adc", fontsize=14)
        ax.set_xlabel("tick", fontsize=14)
        ax.set_xlim(start_tick + 1, end_tick + 1)

        ax2 = ax.twinx()
        ax2.hist(ticks, bins=len(ticks), weights=realA[ch,start_tick:end_tick], histtype='step', label='ND adc', linewidth=0.7, color='g')
        ax2.set_ylabel("ND adc", fontsize=14)

        ax_ylims = ax.axes.get_ylim()
        ax_yratio = ax_ylims[0] / ax_ylims[1]
        ax2_ylims = ax2.axes.get_ylim()
        ax2_yratio = ax2_ylims[0] / ax2_ylims[1]
        if ax_yratio < ax2_yratio:
            ax2.set_ylim(bottom = ax2_ylims[1]*ax_yratio)
        else:
            ax.set_ylim(bottom = ax_ylims[1]*ax2_yratio)

        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles += handles2
        labels += labels2
        new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
        plt.legend(handles=new_handles, labels=labels, prop={'size': 14})

        plt.title("Channel {} in ROP".format(ch), fontsize=16)

        fig.tight_layout()
        pdf2.savefig(bbox_inches='tight')
        plt.close()

    pdf2.close()

    print("mean_L1_loss={}".format(np.mean(losses_pix)))
    print("mean_channel_loss={}".format(np.mean(losses_channel)))

    fig, ax = plt.subplots(figsize=(12,8))
    
    ax.hist(losses_event_over20_fractional, bins=100, range=(-0.5, 0.5), histtype='step')
    ax.set_title("Event over 20 ADC summed", fontsize=16)
    ax.grid(visible=True)
    ax.set_xlabel(r'$\frac{ADC^\prime-ADC}{ADC}$', fontsize=14)
    ax.set_xlim(-0.5, 0.5)

    pdf_bias.savefig(bbox='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(12,8))
    
    ax.hist(losses_event_underneg20_fractional, bins=100, range=(-0.5, 0.5), histtype='step')
    ax.set_title("Event under -20 ADC sum", fontsize=16)
    ax.grid(visible=True)
    ax.set_xlabel(r'$\frac{ADC^\prime-ADC}{ADC}$', fontsize=12)
    ax.set_xlim(-0.5, 0.5)

    pdf_bias.savefig(bbox='tight')
    plt.close()

    pdf_bias.close()

    out_name = "valid_losses_FP16_epoch{}.txt".format(opt.epoch) if half_precision else "valid_losses_epoch{}.txt".format(opt.epoch)
    with open(os.path.join(out_dir, out_name), 'w') as f:
        f.write("mean_L1_loss={}\n".format(np.mean(losses_pix)))
        f.write("mean_channel_loss={}\n".format(np.mean(losses_channel)))
        # f.write("mean_abspix_bias={}\n".format(np.mean(losses_abs_pix_bias))) 
        # f.write("mean_abspix_bias_fractional={}\n".format(np.mean(losses_abs_pix_bias_fractional))) 
        # f.write("mean_channel_bias={}\n".format(np.mean(losses_channel_bias))) 
        # f.write("mean_channel_bias_fractional={}\n".format(np.mean(losses_channel_bias_fractional))) 
        # f.write("mean_event_bias={}\n".format(np.mean(losses_event_bias))) 
        # f.write("mean_event_bias_fractional={}\n".format(np.mean(losses_event_bias_fractional))) 
        f.write("mean_event_over20={}\n".format(np.mean(losses_event_over20)))
        f.write("mean_event_over20_fractional={}\n".format(np.mean(losses_event_over20_fractional)))
        f.write("mean_event_underneg20={}\n".format(np.mean(losses_event_underneg20)))
        f.write("mean_event_underneg20_fractional={}\n".format(np.mean(losses_event_underneg20_fractional)))
        f.write("mean_L1_loss_absover20={}\n".format(np.mean(losses_pix_absover20)))
        f.write("mean_channel_loss_absover20={}\n".format(np.mean(losses_channel_absover20)))

if __name__ == '__main__':
    experiment_dir = '/home/awilkins/extrapolation_pix2pix/checkpoints/nd_fd_radi_1-8_vtxaligned_15'

    with open(os.path.join(experiment_dir, 'config.yaml')) as f:
        options = yaml.load(f, Loader=yaml.FullLoader)

    options['gpu_ids'] = [0]
    # For resnet dropout is in the middle of a sequential so needs to be commented out to maintain layer indices
    # For for unet its at the end so can remove it and still load the state_dict (nn.Dropout has no weights so
    # we don't get an unexpected key error when doing this)
    # options['no_dropout'] = True
    options['num_threads'] = 1
    options['phase'] = 'test'
    options['isTrain'] = False
    options['epoch'] = 'bias_good_mu_best_sigma' # 'latest', 'best_{bias_mu, bias_sigma, loss_pix, loss_channel}', 'bias_good_mu_best_sigma'

    # if options['mask_type'] == 'none_weighted' or options['mask_type'] == 'saved_1rms':
    #     print("How do I want to compare L1 losses for models trained with none_weigthed and saved_time?")
    #     sys.exit()

    # if 'lambda_L1_reg' not in options:
    #     options['lambda_L1_reg'] = 0

    include_realA = True
    
    # True if checkpoint was using nd_fd_radi_1-8_vtxaligned before the nd ped was subtracted
    nd_ped = False

    if 'adam_weight_decay' not in options:
        options['adam_weight_decay'] = 0

    if 'mask_type' not in options:
        if options['using_mask']:
            options['mask_type'] = 'saved'
        else:
            options['mask_type'] = 'none'
        options.pop('using_mask')

    if 'nonzero_L1weight' not in options:
        options['nonzero_L1weight'] = 0

    if 'rms' not in options:
        options['rms'] = 0

    if 'unaligned' not in options:
        options['unaligned'] = False

    half_precision = False
    if half_precision:
        print("###########################################\n" + 
            "Using FP16" +
            "\n###########################################")

    # have replaced valid with a few files of interest
    test_sample = False
    if test_sample:
        print("###########################################\n" + 
            "Using test_sample" +
            "\n###########################################")

    if 'kernel_size' not in options:
        options['kernel_size'] = 4
    if 'outer_stride' not in options:
        options['outer_stride'] = 2
    if 'inner_stride_1' not in options:
        options['inner_stride_1'] = 2

    print("Using configuration:")
    for key, value in options.items():
        print("{}={}".format(key, value))

    print("###########################################\n" +
        "WARNING: bias=use_bias was missing from in_1 unetblock upconv!\n" +
        "some experiments were missing this so will need to remove it manually " +
        "in networks.py when testing them.\n" +
        "Note use_bias=False with batch norm since batch norm has an inbuilt bias term. Since " +
        "batchsize is 1 this batch norm is equivalent to an instance norm but with a bias term " +
        "included.\n" + 
        "WARNING: elif kernel_size == (3,5) and outer_stride == (1,3) and inner_stride_1 == (1,3) " +
        "at L497 had outer_stride == 2 for a long time but still worked (somehow?), " +
        "will need to change this back to == 2 for some models" +
        "\n###########################################")

    MyTuple = namedtuple('MyTuple', options)
    opt = MyTuple(**options)

    main(opt)
