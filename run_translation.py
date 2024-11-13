"""
Run the detector response translation on infilled ND packet projections for each wire plane.
This is part of the detector response translation workflow using the paired hdf5 files.
NOTE It is assumed that some features, like the ch and t slice options, are not used.
"""
import argparse, os, time
from collections import namedtuple
from functools import partialmethod

from matplotlib import pyplot as plt

import numpy as np
import yaml
import h5py
from tqdm import tqdm

import torch

from pix2pix.model import Pix2pix
from data_scripts.make_infilled_pairs import make_nd_pixelmap, make_signalmask

TICK_WINDOW = 4492
DEVICE = torch.device("cuda:0")

def get_options(conf, epoch):
    with open(conf, "r") as f:
        options = yaml.load(f, Loader=yaml.FullLoader)

    # If data is not on the current node, grab it from the share disk.
    if not os.path.exists(options['dataroot']):
        options['dataroot'] = options['dataroot_shared_disk']

    # For eval
    options["num_threads"] = 1
    options["isTrain"] = False

    valid_epochs = {
        "latest",
        "best_bias_mu",
        "best_bias_sigma",
        "best_loss_pix",
        "best_loss_channel",
        "bias_good_mu_best_sigma"
    }
    if epoch not in valid_epochs:
        raise ValueError(f"epoch={epoch} is not valid, use one of {valid_epochs}")
    options["epoch"] = epoch

    print("Using configuration:")
    for key, value in options.items():
        print("{}={}".format(key, value))
    print("------------")

    MyTuple = namedtuple('MyTuple', options)
    opt = MyTuple(**options)

    return opt

def get_model(opt):
    model = Pix2pix(opt)
    print("model {} was created".format(type(model).__name__))
    model.setup(opt)
    model.eval()
    return model

def get_h5_files(input_file, output_file, drop_3d, drop_projs):
    f_in = h5py.File(input_file)
    f_out = h5py.File(output_file, "w")

    # Starting at root key "/", recursively flatten the key tree
    def find_h5_keys(h5_obj):
        keys = tuple()
        if isinstance(h5_obj, h5py.Group):
            for val in h5_obj.values():
                if isinstance(h5_obj, h5py.Group):
                    keys += find_h5_keys(val)
                else:
                    keys += (val.name,)
        elif isinstance(h5_obj, h5py.Dataset):
            keys += (h5_obj.name,)
        return keys
    dataset_keys = find_h5_keys(f_in["/"])

    # Copy over data to new h5
    for key in dataset_keys:
        if (
            (drop_3d and key.startswith("/3d_packets_infilled")) or
            (drop_projs and key.startswith("/nd_packet_wire_projs"))
        ):
            continue
        data = np.array(f_in[key])
        f_out.create_dataset(key, data=data)

    return f_in, f_out

def get_plane_data(rop, args):
    if rop == "2" or rop == "3":
        return (
            "Z",
            (480, TICK_WINDOW),
            args.signalmask_max_tick_negative_Z,
            args.signalmask_max_tick_positive_Z,
            args.signalmask_max_ch_Z,
            args.threshold_mask_Z
        )
    if rop == "1":
        return (
            "U",
            (800, TICK_WINDOW),
            args.signalmask_max_tick_negative_U,
            args.signalmask_max_tick_positive_U,
            args.signalmask_max_ch_U,
            args.threshold_mask_U
        )
    if rop == "0":
        return (
            "V",
            (800, TICK_WINDOW),
            args.signalmask_max_tick_negative_V,
            args.signalmask_max_tick_positive_V,
            args.signalmask_max_ch_V,
            args.threshold_mask_V
        )

def prep_model_input(opt, nd_arr, signal_mask):
    # Repeat what the dataloader would be doing
    for i in range(opt.input_nc):
        nd_arr[i] *= opt.A_ch_scalefactors[i]
    A = torch.from_numpy(nd_arr).float()
    mask = torch.from_numpy(np.expand_dims(signal_mask, axis=0)[:, :, :]).float()

    # Give batch dimension
    A = A.unsqueeze(0)
    mask = mask.unsqueeze(0)
    A = A.to(DEVICE)
    mask = mask.to(DEVICE)

    # Return in same format as the dataloader
    return {"A" : A, "A_paths" : "null", "B_path" : "null", "mask" : mask }

def make_threshold_mask(pred_arr, threshold_val):
    mask = np.abs(pred_arr)
    mask *= (mask > threshold_val)
    # Smear a bit to let signal ADC drop gracefully
    for _ in range(1, 6 + 1): # Smear by 5 ticks
        mask[:, 1:] += mask[:, :-1]
        mask[:, :-1] += mask[:, 1:]
    for _ in range(1, 1 + 1): # Smear by one channel
        mask[1:, :] += mask[:-1, :]
        mask[:-1, :] += mask[1:, :]
    return mask.astype(bool).astype(int)

def main(args):
    opt_Z = get_options(args.config_Z, args.epoch_Z)
    opt_U = get_options(args.config_U, args.epoch_U)
    opt_V = get_options(args.config_V, args.epoch_V)
    opts = { "Z" : opt_Z, "U" : opt_U, "V" : opt_V }

    model_Z = get_model(opt_Z)
    model_U = get_model(opt_U)
    model_V = get_model(opt_V)
    models = { "Z" : model_Z, "U" : model_U, "V" : model_V }

    f_in, f_out = get_h5_files(args.input_file, args.output_file, args.drop_3d, args.drop_projs)

    start = time.time()

    tot_evids = len(f_in["nd_packet_wire_projs"])
    for i_evid, evid in enumerate(f_in["nd_packet_wire_projs"].keys()):
        print(f"{i_evid} / {tot_evids}")
        t_pms, t_smasks, t_infs, t_outs, t_thres, t_writes = [], [], [], [], [], []
        for tpcset in f_in["nd_packet_wire_projs"][evid].keys():
            for rop in f_in["nd_packet_wire_projs"][evid][tpcset].keys():
                ret = get_plane_data(rop, args)
                signal_type = ret[0]
                plane_shape = ret[1]
                signalmask_max_tick_negative, signalmask_max_tick_positive = ret[2], ret[3]
                signalmask_max_ch = ret[4]
                threshold_val = ret[5]
                opt = opts[signal_type]
                model = models[signal_type]

                t_0 = time.time()
                nd_arr = make_nd_pixelmap(
                    f_in["nd_packet_wire_projs"][evid][tpcset][rop], plane_shape, False
                )
                t_pms.append(time.time() - t_0)
                t_0 = time.time()
                signal_mask = make_signalmask(
                    nd_arr,
                    signalmask_max_tick_positive, signalmask_max_tick_negative, signalmask_max_ch
                )
                t_smasks.append(time.time() - t_0)
                t_0 = time.time()

                input_data = prep_model_input(opt, nd_arr, signal_mask)

                model.set_input(input_data, test=True)
                model.test()
                t_infs.append(time.time() - t_0)
                t_0 = time.time()

                vis = model.get_current_visuals()
                pred = vis["fake_B"].cpu()
                mask = input_data["mask"].cpu()
                pred *= mask # XXX important, zeroing out predicted response outside of signal mask.
                pred = pred[0][0] # remove batch and adc channel dimensions
                pred /= opt.B_ch_scalefactors[0]
                pred = pred.numpy().astype(int)
                t_outs.append(time.time() - t_0)
                t_0 = time.time()

                # Trying to zero out regions that are probably noise. Found this is especially
                # important for the induction plane models where there are long tails of very low
                # ADC that confuse the signal processing and produce very wide hits. I think the
                # signal processing is expecting the signal to cross zero at some point. The
                # tanh at the end of the induction plane models makes generating a zero difficult,
                # unlike the collection plane models.
                if threshold_val is not None:
                    threshold_mask = make_threshold_mask(pred, threshold_val)
                    pred *= threshold_mask
                t_thres.append(time.time() - t_0)
                t_0 = time.time()

                if args.make_plots:
                    if not os.path.exists("validation_plots"):
                        os.makedirs("validation_plots")
                    _, ax = plt.subplots(1, 2, figsize=(12, 6))
                    vmax = np.max(np.abs(nd_arr[0]))
                    vmin = -vmax
                    nd_arr[0][nd_arr[5] == 1] = -nd_arr[0][nd_arr[5] == 1]
                    ax[0].imshow(
                        np.ma.masked_where(nd_arr[0] == 0, nd_arr[0]).T,
                        origin="lower",
                        aspect="auto",
                        interpolation="none",
                        cmap="seismic",
                        vmin=vmin, vmax=vmax
                    )
                    vmax = np.max(np.abs(pred))
                    vmin = -vmax
                    ax[1].imshow(
                        pred.T,
                        origin="lower",
                        aspect="auto",
                        interpolation="none",
                        cmap="seismic",
                        vmin=vmin, vmax=vmax
                    )
                    plt.savefig(os.path.join("validation_plots", f"{evid}-{tpcset}-{rop}.pdf"))
                    plt.close()

                f_out.create_dataset(
                    "/pred_fd_resps/" + evid + "/" + tpcset + "/" + rop,
                    data=pred, compression="gzip", compression_opts=9
                )
                t_writes.append(time.time() - t_0)
        # print(f"pixelmap = {np.sum(t_pms):.2f}s")
        # print(f"signalmask = {np.sum(t_smasks):.2f}s")
        # print(f"inference = {np.sum(t_infs):.2f}s")
        # print(f"output = {np.sum(t_outs):.2f}s")
        # print(f"threshold = {np.sum(t_thres):.2f}s")
        # print(f"write = {np.sum(t_writes):.2f}s")
        # print(f"total = {np.sum(t_pms + t_smasks + t_infs + t_outs + t_thres + t_writes)}s")
        # print("-------")

    end = time.time()
    delta = end - start
    print(f"Finished in {delta / 60**2:.3f} hrs, average {delta / tot_evids:.3f} s per event")

    f_out.close()
    f_in.close()

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("config_Z", type=str, help="Experiment config yaml for Z plane model")
    parser.add_argument("epoch_Z", type=str, help="Epoch of saved model for Z plane")
    parser.add_argument("config_U", type=str, help="Experiment config yaml for U plane model")
    parser.add_argument("epoch_U", type=str, help="Epoch of saved model for U plane")
    parser.add_argument("config_V", type=str, help="Experiment config yaml for V plane model")
    parser.add_argument("epoch_V", type=str, help="Epoch of saved model for V plane")

    parser.add_argument(
        "--signalmask_max_tick_positive_Z",
        type=int, default=30,
        help="Max ND packet smearing in positive tick direction to make Z plane signal mask."
    )
    parser.add_argument(
        "--signalmask_max_tick_negative_Z",
        type=int, default=30,
        help="Max ND packet smearing in negative tick direction to make Z plane signal mask."
    )
    parser.add_argument(
        "--signalmask_max_ch_Z",
        type=int, default=4,
        help="Max ND packet smearing in channel direction to make Z plane signal mask."
    )
    parser.add_argument(
        "--signalmask_max_tick_positive_U",
        type=int, default=50,
        help="Max ND packet smearing in positive tick direction to make U plane signal mask."
    )
    parser.add_argument(
        "--signalmask_max_tick_negative_U",
        type=int, default=50,
        help="Max ND packet smearing in negative tick direction to make U plane signal mask."
    )
    parser.add_argument(
        "--signalmask_max_ch_U",
        type=int, default=4,
        help="Max ND packet smearing in channel direction to make U plane signal mask."
    )
    parser.add_argument(
        "--signalmask_max_tick_positive_V",
        type=int, default=50,
        help="Max ND packet smearing in positive tick direction to make V plane signal mask."
    )
    parser.add_argument(
        "--signalmask_max_tick_negative_V",
        type=int, default=50,
        help="Max ND packet smearing in negative tick direction to make V plane signal mask."
    )
    parser.add_argument(
        "--signalmask_max_ch_V",
        type=int, default=4,
        help="Max ND packet smearing in channel direction to make V plane signal mask."
    )

    parser.add_argument(
        "--drop_3d", action="store_true",
        help="Drop the '3d_packets_infilled' group from output hdf5"
    )
    parser.add_argument(
        "--drop_projs", action="store_true",
        help="Drop the 'nd_packet_wire_projs' group from the output hdf5"
    )

    parser.add_argument(
        "--make_plots", action="store_true",
        help="Save validation plots to validation_plots/"
    )

    parser.add_argument(
        "--threshold_mask_Z", type=int, default=None,
        help="Set ADCs below this threshold to zero (with small amount of smearing)"
    )
    parser.add_argument(
        "--threshold_mask_U", type=int, default=None,
        help="Set ADCs below this threshold to zero (with small amount of smearing)"
    )
    parser.add_argument(
        "--threshold_mask_V", type=int, default=None,
        help="Set ADCs below this threshold to zero (with small amount of smearing)"
    )

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    args = parse_arguments()
    main(args)

