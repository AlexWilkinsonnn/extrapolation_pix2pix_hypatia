import argparse, time, os, shutil
from collections import namedtuple

import numpy as np
from scipy import stats
import yaml

from pix2pix.model import Pix2pix
from pix2pix.dataset import CustomDatasetDataLoader
from pix2pix.losses import CustomLoss

# import torch; torch.autograd.set_detect_anomaly(True)

def main(opt):
    dataset = CustomDatasetDataLoader(opt).load_data()
    dataset_size = len(dataset)
    print("Number of training images = %d" % dataset_size)

    dataset_valid = CustomDatasetDataLoader(opt, valid=True).load_data()
    dataset_valid_size = len(dataset_valid)
    print("Number of validation images = %d" % dataset_valid_size)
    dataset_valid_iterator = iter(dataset_valid)

    model = Pix2pix(opt)
    print("model [%s] was created" % type(model).__name__)
    model.setup(opt)
    if hasattr(opt, "load_from_epoch"):
        model.load_networks(
            os.path.basename(opt.load_from_epoch),
            networks_dir=os.path.dirname(opt.load_from_epoch)
        )
    n_iter = 0

    best_metrics = {}

    with open(os.path.join(opt.checkpoints_dir, opt.name, "loss.txt"), 'a') as f:
        f.write("Iters per epoch: {}\n".format(len(dataset.dataloader)))

    for epoch in range(opt.n_epochs + opt.n_epochs_decay):
        with open(os.path.join(opt.checkpoints_dir, opt.name, "loss.txt"), 'a') as f:
            f.write("==== Epoch {} ====\n".format(epoch))

        epoch_start_time = time.time()

        for n_iter_epoch, data in enumerate(dataset):
            data['mask'].requires_grad = False

            model.set_input(data)
            model.optimize_parameters()

            if (n_iter + 1) % opt.display_freq == 0 or (n_iter + 1) % opt.print_freq == 0:
                losses = model.get_current_losses()
                loss_line = (
                    "Epoch: {} Iter: {} Total Iter: {} -- G_GAN={:.5f} G_pix={:.5f} "
                    "G_channel={:.5f} D_real={:.5f} D_fake={:.5f}"
                ).format(
                    epoch, n_iter_epoch, n_iter, losses['G_GAN'], losses['G_pix'],
                    losses['G_channel'], losses['D_real'], losses['D_fake']
                )
                with open(os.path.join(opt.checkpoints_dir, opt.name, "loss.txt"), 'a') as f:
                    f.write(loss_line + '\n')

                if (n_iter + 1) % opt.print_freq == 0:
                    print(loss_line)

                if (n_iter + 1) % opt.display_freq == 0:
                    visuals = model.get_current_visuals()

                    image_realA = visuals['real_A'][0].data # Taking first in the batch to save
                    image_realA[0] /= opt.A_ch_scalefactors[0]
                    arr_realA = image_realA.cpu().float().numpy()
                    arr_realA[0] = arr_realA[0].astype(int)
                    np.save(os.path.join(opt.checkpoints_dir, opt.name, "realA.npy"), arr_realA[0])

                    image_realB = visuals['real_B'][0].data
                    image_realB[0] /= opt.B_ch_scalefactors[0]
                    arr_realB = image_realB.cpu().float().numpy().astype(int)
                    np.save(os.path.join(opt.checkpoints_dir, opt.name, "realB.npy"), arr_realB[0])

                    image_fakeB = visuals['fake_B'][0].data
                    image_fakeB[0] /= opt.B_ch_scalefactors[0]
                    arr_fakeB = image_fakeB.cpu().float().numpy().astype(int)
                    np.save(os.path.join(opt.checkpoints_dir, opt.name, "fakeB.npy"), arr_fakeB[0])

            # cache our latest model every <save_latest_freq> iterations
            if (n_iter + 1) % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, n_iter))
                save_suffix = 'latest'
                model.save_networks(save_suffix)

            if opt.valid_freq != 'epoch' and n_iter % opt.valid_freq == 0:
                valid(
                    dataset_valid_iterator, dataset_valid, model, opt, epoch, n_iter, best_metrics
                )

            n_iter += 1

        model.update_learning_rate()

        if (epoch + 1) % opt.save_epoch_freq == 0: # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, total iters %d' % (epoch, n_iter))
            model.save_networks('latest')
            model.save_networks(epoch)

        print(
            "End of epoch {} / {} \t Time Taken: {} sec".format(
                epoch + 1, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time
            )
        )

        if opt.valid_freq == 'epoch':
            valid(dataset_valid_iterator, dataset_valid, model, opt, epoch, n_iter, best_metrics)


def valid(dataset_itr, dataset, model, opt, epoch, total_itrs, best_metrics):
    with open(os.path.join(opt.checkpoints_dir, opt.name, "loss.txt"), 'a') as f:
        f.write("== Validation Loop ==\n")

    model.eval()
    dataset_itr = iter(dataset)

    G_pix_losses, G_channel_losses = [], []
    G_pix_losses_orig, G_channel_losses_orig = [], []
    losses_event_over20, losses_event_over20_fractional = [], []
    losses_event_underneg20, losses_event_underneg20_fractional = [], []
    for i in range(len(dataset)): # Run the generator on some validation images for visualisation
        if opt.num_valid and i > opt.num_valid:
            break

        try:
            data = next(dataset_itr)
        except StopIteration:
            dataset_itr = iter(dataset)
            data = next(dataset_itr)

        model.set_input(data)
        model.test()

        losses_orig = model.get_current_losses()
        G_pix_losses_orig.append(losses_orig["G_pix"])
        G_channel_losses_orig.append(losses_orig["G_channel"])

        visuals = model.get_current_visuals()

        if opt.channel_offset != 0:
            ch_slicel, ch_sliceh = opt.channel_offset, -opt.channel_offset
        else:
            ch_slicel, ch_sliceh = None, None
        if opt.tick_offset != 0:
            t_slicel, t_sliceh = opt.tick_offset, -opt.tick_offset
        else:
            t_slicel, t_sliceh = None, None
        realA = visuals['real_A'].cpu()[:, :, ch_slicel:ch_sliceh, t_slicel:t_sliceh]
        realB = visuals['real_B'].cpu()[:, :, ch_slicel:ch_sliceh, t_slicel:t_sliceh]
        fakeB = visuals['fake_B'].cpu()[:, :, ch_slicel:ch_sliceh, t_slicel:t_sliceh]
        mask = data['mask'].cpu()[:, :, ch_slicel:ch_sliceh, t_slicel:t_sliceh]
        fakeB *= mask
        realB /= opt.B_ch_scalefactors[0]
        fakeB /= opt.B_ch_scalefactors[0]

        loss_pix, loss_channel = CustomLoss(
            realA, fakeB, realB,
            mask,
            opt.B_ch_scalefactors[0],
            opt.mask_type,
            opt.nonzero_L1weight,
            opt.rms
        )
        loss_event_over20 = (
            (fakeB.float() * (fakeB.float() > 20)).sum() -
            (realB.float() * (realB.float() > 20)).sum()
        )
        loss_event_over20_fractional = (
            loss_event_over20 / ((realB.float() * (realB.float() > 20)).sum())
        )

        if realA.shape[2] == 800: # induction view
            loss_event_underneg20 = (
                (fakeB.float() * (fakeB.float() < -20)).sum() -
                (realB.float() * (realB.float() < -20)).sum()
            )
            loss_event_underneg20_fractional = (
                loss_event_underneg20 / ((realB.float() * (realB.float() < -20)).sum())
            )

        if loss_pix != 0:
            G_pix_losses.append(loss_pix.item())
            G_channel_losses.append(loss_channel.item())
            if (realB.float() * (realB.float() > 20)).sum() != 0:
                losses_event_over20.append(loss_event_over20.item())
                losses_event_over20_fractional.append(loss_event_over20_fractional.item())
            if realA.shape[2] == 800 and (realB.float() * (realB.float() < -20)).sum() != 0:
                losses_event_underneg20.append(loss_event_underneg20.item())
                losses_event_underneg20_fractional.append(loss_event_underneg20_fractional.item())

        if i < 10:
            realA = realA[0] # Remove batch dimension (batchsize = 1 for valid)
            realA[0] /= opt.A_ch_scalefactors[0]
            arr_realA = realA.numpy()
            arr_realA[0] = arr_realA[0].astype(int)
            np.save(
                os.path.join( opt.checkpoints_dir, opt.name, "realA_valid{}.npy".format(i)),
                arr_realA[0]
            )

            arr_realB = realB[0].numpy().astype(int)
            np.save(
                os.path.join(opt.checkpoints_dir, opt.name, "realB_valid{}.npy".format(i)),
                arr_realB[0]
            )

            arr_fakeB =fakeB[0].numpy().astype(int)
            np.save(
                os.path.join(opt.checkpoints_dir, opt.name, "fakeB_valid{}.npy".format(i)),
                arr_fakeB[0]
            )

    if realA.shape[2] == 800: # induction view
        bias_mu_over20, bias_sigma_over20 = stats.norm.fit(losses_event_over20_fractional)
        bias_mu_underneg20, bias_sigma_underneg20 = stats.norm.fit(
            losses_event_underneg20_fractional
        )
        bias_mu = (abs(float(bias_mu_over20)) + abs(float(bias_mu_underneg20))) / 2
        bias_sigma = (abs(float(bias_sigma_over20)) + abs(float(bias_sigma_underneg20))) / 2

    else:
        bias_mu, bias_sigma = stats.norm.fit(losses_event_over20_fractional)
        bias_mu, bias_sigma = float(bias_mu), float(bias_sigma)

    loss_pix, loss_channel = float(np.mean(G_pix_losses)), float(np.mean(G_channel_losses))

    if 'bias_mu' not in best_metrics or abs(best_metrics['bias_mu']) > abs(bias_mu):
        best_metrics['bias_mu'] = bias_mu
        best_metrics['bias_mu_itr'] = total_itrs
        best_metrics['bias_mu_epoch'] = epoch
        model.save_networks("best_bias_mu")

    if 'bias_sigma' not in best_metrics or best_metrics['bias_sigma'] > bias_sigma:
        best_metrics['bias_sigma'] = bias_sigma
        best_metrics['bias_sigma_itr'] = total_itrs
        best_metrics['bias_sigma_epoch'] = epoch
        model.save_networks("best_bias_sigma")

    if abs(bias_mu) < abs(0.05):
        if (
            'bias_good_mu_best_sigma' not in best_metrics or
            best_metrics['bias_good_mu_best_sigma'][1] > bias_sigma
        ):
            best_metrics['bias_good_mu_best_sigma'] = (bias_mu, bias_sigma)
            best_metrics['bias_good_mu_best_sigma_itr'] = total_itrs
            best_metrics['bias_good_mu_best_sigma_epoch'] = epoch
            model.save_networks("bias_good_mu_best_sigma")

    if 'loss_pix' not in best_metrics or best_metrics['loss_pix'] > loss_pix:
        best_metrics['loss_pix'] = loss_pix
        best_metrics['loss_pix_itr'] = total_itrs
        best_metrics['loss_pix_epoch'] = epoch
        model.save_networks("best_loss_pix")

    if 'loss_channel' not in best_metrics or best_metrics['loss_channel'] > loss_channel:
        best_metrics['loss_channel'] = loss_channel
        best_metrics['loss_channel_itr'] = total_itrs
        best_metrics['loss_channel_epoch'] = epoch
        model.save_networks("best_loss_channel")

    print(best_metrics)

    with open(
        os.path.join(options['checkpoints_dir'], options['name'], "best_metrics.yaml"), 'w'
    ) as f:
        yaml.dump(best_metrics, f)

    loss_line = (
        "VALID: Epoch: {} Total Iter: {} -- "
        "G_pix={} G_channel={} "
        "G_pix_unscaled={} G_channel_unscaled={}"
    ).format(
        epoch, total_itrs,
        np.mean(G_pix_losses_orig), np.mean(G_channel_losses_orig),
        np.mean(G_pix_losses), np.mean(G_channel_losses)
    )
    print(loss_line)
    with open(os.path.join(opt.checkpoints_dir, opt.name, "loss.txt"), 'a') as f:
        f.write(loss_line + '\n')

    model.train()


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("config")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()
    with open(args.config, "r") as f:
        options = yaml.load(f, Loader=yaml.FullLoader)

    # If data is not on the current node, grab it from the share disk.
    if not os.path.exists(options['dataroot']):
        options['dataroot'] = options['dataroot_shared_disk']

    print("Using configuration:")
    for key, value in options.items():
        print("{}={}".format(key, value))

    if not os.path.exists(os.path.join(options['checkpoints_dir'], options['name'])):
        os.makedirs(os.path.join(options['checkpoints_dir'], options['name']))

    shutil.copyfile(
        args.config, os.path.join(options['checkpoints_dir'], options['name'], "config.yaml")
    )

    if options['noise_layer']:
        options['input_nc'] += 1

    MyTuple = namedtuple('MyTuple', options)
    opt = MyTuple(**options)

    main(opt)

