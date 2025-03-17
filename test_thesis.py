import argparse, os, sys
from collections import namedtuple
from operator import itemgetter

import numpy as np
import yaml
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

import torch

from pix2pix.model import Pix2pix
from pix2pix.dataset import CustomDatasetDataLoader
from pix2pix.losses import CustomLoss

OCCUPIED_THRESHOLD = 20
INDUCTION = True
NOISE_ONLY = False

def main(opt):
    out_dir = os.path.join(
        '/home/awilkins/extrapolation_pix2pix/results',
        os.path.join(os.path.basename(opt.dataroot), opt.name, opt.epoch)
    )

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dataset_test = CustomDatasetDataLoader(opt, test=True).load_data()
    dataset_test_size = len(dataset_test)
    print("Number of test images={}".format(dataset_test_size))

    model = Pix2pix(opt)
    print("model {} was created".format(type(model).__name__))
    model.setup(opt)
    model.eval()

    if opt.channel_offset != 0:
        ch_slicel, ch_sliceh = opt.channel_offset, -opt.channel_offset
    else:
        ch_slicel, ch_sliceh = None, None
    if opt.tick_offset != 0:
        t_slicel, t_sliceh = opt.tick_offset, -opt.tick_offset
    else:
        t_slicel, t_sliceh = None, None

    losses_pix, losses_channel = [], []
    purities, completenesses = [], []
    channel_adc_diffs_pos, channel_adc_diffs_neg = [], []
    channel_adc_diffs, channel_occupied_diffs = [], []
    event_adc_diffs, event_occupied_diffs = [], []

    if opt.half_precision:
        name = 'output_images_FP16_epoch{}.pdf'.format(opt.epoch)
    else:
        name = 'output_images_epoch{}.pdf'.format(opt.epoch)
    if test_sample:
        name = 'test_sample_' + name
    pdf = PdfPages(os.path.join(out_dir, name))
    if opt.half_precision:
        name_bias = 'output_biashist_FP16_epoch{}.pdf'.format(opt.epoch)
    else:
        name_bias = 'output_biashist_epoch{}.pdf'.format(opt.epoch)
    if test_sample:
        name_bias = 'test_sample_' + name_bias

    if NOISE_ONLY:
        noise_mae = []
        dataset_test_iter = iter(dataset_test)
        data_curr = next(dataset_test_iter)
        for i in range(999999):
            # if i == 500:
            #     break

            if (i % 1000 == 0):
                print(i)

            try:
                data_next = next(dataset_test_iter)
            except StopIteration:
                break

            model.set_input(data_curr)
            visuals_curr = model.get_current_visuals()
            realB_curr = visuals_curr['real_B'].cpu()[:, :, ch_slicel:ch_sliceh, t_slicel:t_sliceh]
            mask_curr = data_curr['mask'].cpu()[:, :, ch_slicel:ch_sliceh, t_slicel:t_sliceh]
            realB_curr /= opt.B_ch_scalefactors[0]

            model.set_input(data_next)
            visuals_next = model.get_current_visuals()
            realB_next = visuals_next['real_B'].cpu()[:, :, ch_slicel:ch_sliceh, t_slicel:t_sliceh]
            mask_next = data_next['mask'].cpu()[:, :, ch_slicel:ch_sliceh, t_slicel:t_sliceh]
            realB_next /= opt.B_ch_scalefactors[0]

            mask_both = ~(mask_curr.bool()) * ~(mask_next.bool())

            loss_pix, _ = CustomLoss(
                None, realB_curr.float(), realB_next.float(),
                mask_both.float(),
                opt.B_ch_scalefactors[0],
                opt.mask_type,
                opt.nonzero_L1weight,
                opt.rms
            )
            noise_mae.append(loss_pix)

            data_curr = data_next

        print(np.mean(noise_mae))
        np.save(os.path.join(out_dir, f"thesis_plots/noise_losses_pix.npy"),noise_mae)

        return

    for i, data in enumerate(dataset_test):
        # if i == 500:
        #     break

        if (i % 1000 == 0):
            print(i)

        model.set_input(data)
        model.test(opt.half_precision)
        
        visuals = model.get_current_visuals()
        realA = visuals['real_A'].cpu()[:, :, ch_slicel:ch_sliceh, t_slicel:t_sliceh]
        realB = visuals['real_B'].cpu()[:, :, ch_slicel:ch_sliceh, t_slicel:t_sliceh]
        fakeB = visuals['fake_B'].cpu()[:, :, ch_slicel:ch_sliceh, t_slicel:t_sliceh]
        mask = data['mask'].cpu()[:, :, ch_slicel:ch_sliceh, t_slicel:t_sliceh]
        fakeB *= mask
        realA /= opt.A_ch_scalefactors[0]
        realB /= opt.B_ch_scalefactors[0]
        fakeB /= opt.B_ch_scalefactors[0]
        loss_pix, loss_channel = CustomLoss(
            realA.float(), fakeB.float(), realB.float(),
            mask.float(),
            opt.B_ch_scalefactors[0],
            opt.mask_type,
            opt.nonzero_L1weight,
            opt.rms
        )
        losses_pix.append(loss_pix)
        losses_channel.append(loss_channel)

        continue # XXX

        realA_infill = realA[0][5]
        realA = realA[0][0]
        realB = realB[0][0]
        fakeB = fakeB[0][0]

        if INDUCTION:
            realB *= mask[0][0]

            for ch in range(len(realB)):
                if not torch.any(realB[ch]):
                    continue

                true_sum_pos = torch.sum(realB[ch] * (realB[ch] >= 0)).item()
                pred_sum_pos = torch.sum(fakeB[ch] * (fakeB[ch] >= 0)).item()
                if pred_sum_pos == true_sum_pos:
                    channel_adc_diffs_pos.append(0.0)
                else:
                    if true_sum_pos == 0.0:
                        true_sum_pos = 1.0
                    channel_adc_diffs_pos.append((true_sum_pos - pred_sum_pos) / true_sum_pos)
                true_sum_neg = torch.sum(realB[ch] * (realB[ch] < 0)).item()
                pred_sum_neg = torch.sum(fakeB[ch] * (fakeB[ch] < 0)).item()
                if pred_sum_neg == true_sum_neg:
                    channel_adc_diffs_neg.append(0.0)
                else:
                    if true_sum_neg == 0.0:
                        true_sum_neg = 1.0
                    channel_adc_diffs_neg.append((true_sum_neg - pred_sum_neg) / true_sum_neg)
        else:
            realB *= mask[0][0]

            # true_occupied = (realB > OCCUPIED_THRESHOLD)
            # pred_occupied = (fakeB > OCCUPIED_THRESHOLD)
            # tp = torch.sum(true_occupied * pred_occupied).item()
            # fp = torch.sum(~true_occupied * pred_occupied).item()
            # fn = torch.sum(true_occupied * ~pred_occupied).item()
            # if tp == 0:
            #     if fp == 0:
            #         purities.append(1.0)
            #     else:
            #         purities.append(0.0)
            #     if fn == 0:
            #         completenesses.append(1.0)
            #     else:
            #         completenesses.append(0.0)
            # else:
            #     purities.append(tp / (tp + fp))
            #     completenesses.append(tp / (tp + fn))

            for ch in range(len(realB)):
                if not torch.any(realB[ch]):
                    continue

                true_sum, pred_sum = torch.sum(realB[ch]).item(), torch.sum(fakeB[ch]).item()
                if pred_sum == true_sum:
                    channel_adc_diffs.append(0.0)
                else:
                    if true_sum == 0.0:
                        true_sum = 1.0
                    channel_adc_diffs.append((true_sum - pred_sum) / true_sum)
                # true_sum, pred_sum = torch.sum(true_occupied[ch]).item(), torch.sum(pred_occupied[ch]).item()
                # if pred_sum == true_sum:
                #     channel_occupied_diffs.append(0.0)
                # else:
                #     if true_sum == 0:
                #         true_sum = 1
                #     channel_occupied_diffs.append((true_sum - pred_sum) / true_sum)

            # true_sum, pred_sum = torch.sum(realB).item(), torch.sum(fakeB).item()
            # if true_sum == pred_sum:
            #     event_adc_diffs.append(0.0)
            # else:
            #     if true_sum == 0.0:
            #         true_sum = 1.0
            #     event_adc_diffs.append((true_sum - pred_sum) / true_sum)
            # true_sum, pred_sum = torch.sum(true_occupied).item(), torch.sum(pred_occupied).item()
            # if true_sum == pred_sum:
            #     event_occupied_diffs.append(0.0)
            # else:
            #     if true_sum == 0:
            #         true_sum = 1
            #     event_occupied_diffs.append((true_sum - pred_sum) / true_sum)

        if i < 20:
            np.save(os.path.join(out_dir, f"thesis_plots/{i}_nd_infill.npy"), realA_infill)
            np.save(os.path.join(out_dir, f"thesis_plots/{i}_nd.npy"), realA)
            np.save(os.path.join(out_dir, f"thesis_plots/{i}_truefd.npy"), realB)
            np.save(os.path.join(out_dir, f"thesis_plots/{i}_predfd.npy"), fakeB)

    print(np.mean(losses_pix))
    print(np.mean(losses_channel))
    # print(np.mean(purities))
    # print(np.mean(completenesses))
    print(np.mean(channel_adc_diffs), np.median(channel_adc_diffs))
    if INDUCTION:
        print(np.mean(channel_adc_diffs_pos), np.median(channel_adc_diffs_pos))
        print(np.mean(channel_adc_diffs_neg), np.median(channel_adc_diffs_neg))
    # print(np.mean(channel_occupied_diffs), np.median(channel_occupied_diffs))
    # print(np.mean(event_adc_diffs), np.median(event_adc_diffs))
    # print(np.mean(event_occupied_diffs), np.median(event_occupied_diffs))

    np.save(os.path.join(out_dir, f"thesis_plots/losses_pix.npy"), losses_pix)
    np.save(os.path.join(out_dir, f"thesis_plots/losses_channel.npy"), losses_channel)
    # np.save(os.path.join(out_dir, f"thesis_plots/purities.npy"), purities)
    # np.save(os.path.join(out_dir, f"thesis_plots/completenesses.npy"), completenesses)
    np.save(os.path.join(out_dir, f"thesis_plots/channel_adc_diffs.npy"), channel_adc_diffs)
    if INDUCTION:
        np.save(os.path.join(out_dir, f"thesis_plots/channel_adc_diffs_pos.npy"), channel_adc_diffs_pos)
        np.save(os.path.join(out_dir, f"thesis_plots/channel_adc_diffs_neg.npy"), channel_adc_diffs_neg)
    # np.save(os.path.join(out_dir, f"thesis_plots/channel_occupied_diffs.npy"), channel_occupied_diffs)
    # np.save(os.path.join(out_dir, f"thesis_plots/event_adc_diffs.npy"), event_adc_diffs)
    # np.save(os.path.join(out_dir, f"thesis_plots/event_occupied_diffs.npy"), event_occupied_diffs)

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("config")

    parser.add_argument("--no_dropout", type=str, default="")
    parser.add_argument("--epoch", type=str, default="")
    parser.add_argument("--half_precision", action="store_true")
    parser.add_argument("--test_sample", action="store_true")

    parser.add_argument("--induction", action="store_true")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()
    with open(args.config, "r") as f:
        options = yaml.load(f, Loader=yaml.FullLoader)

    # If data is not on the current node, grab it from the share disk.
    if not os.path.exists(options['dataroot']):
        options['dataroot'] = options['dataroot_shared_disk']

    # For resnet dropout is in the middle of a sequential so needs to be commented out to maintain
    # layer indices
    # For for unet its at the end so can remove it and still load the state_dict
    # (nn.Dropout has no weights so we don't get an unexpected key error when doing this)
    if args.no_dropout:
        options["no_dropout"] = args.no_dropout
    options["num_threads"] = 1
    options["isTrain"] = False

    if args.epoch not in [
        "", "latest", "best_bias_mu", "best_bias_sigma", "best_loss_pix", "best_loss_channel",
        "bias_good_mu_best_sigma"
    ]:
        raise ValueError("epoch={} is not valid".format(args.epoch))
    options["epoch"] = args.epoch if args.epoch else "latest"

    if args.half_precision:
        print(
            "###########################################\n" +
            "Using FP16" +
            "\n###########################################"
        )
    options["half_precision"] = args.half_precision

    # have replaced valid with a few files of interest
    test_sample = False
    if test_sample:
        print(
            "###########################################\n" +
            "Using test_sample" +
            "\n###########################################"
        )

    if options['noise_layer']:
        options['input_nc'] += 1

    print("Using configuration:")
    for key, value in options.items():
        print("{}={}".format(key, value))

    # Some old warnings that may not be relevant anymore
    # WARNING: bias=use_bias was missing from in_1 unetblock upconv!\n +
    # some experiments were missing this so will need to remove it manually  +
    # in networks.py when testing them.\n +
    # Note use_bias=False with batch norm since batch norm has an inbuilt bias term. Since  +
    # batchsize is 1 this batch norm is equivalent to an instance norm but with a bias term  +
    # included.\n +
    # WARNING: elif kernel_size == (3,5) and outer_stride == (1,3) and inner_stride_1 == (1,3)  +
    # at L497 had outer_stride == 2 for a long time but still worked (somehow?),  +
    # will need to change this back to == 2 for some models +

    MyTuple = namedtuple('MyTuple', options)
    opt = MyTuple(**options)

    main(opt)

