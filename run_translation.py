"""
Run the detector response translation on infilled ND packet projections for each wire plane.
This is part of the detector response translation workflow using the paired hdf5 files.
NOTE It is assumed that some features, like the ch and t slice options, are not used.
"""
import argparse, os, sys
from collections import namedtuple
from functools import partialmethod

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
        return keys
    dataset_keys = find_h5_keys(f_out["/"])
    print(dataset_keys)

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
            args.signalmask_max_ch_Z
        )
    if rop == "1":
        return (
            "U",
            (800, TICK_WINDOW),
            args.signalmask_max_tick_negative_U,
            args.signalmask_max_tick_positive_U,
            args.signalmask_max_ch_U
        )
    if rop == "0":
        return (
            "V",
            (800, TICK_WINDOW),
            args.signalmask_max_tick_negative_V,
            args.signalmask_max_tick_positive_V,
            args.signalmask_max_ch_V
        )

def prep_model_input(opt, nd_arr, signal_mask):
    # Repeat what the dataloader would be doing
    for i in range(opt.input_nc):
        nd_arr[i] *= opt.A_ch_scalefactors[i]
    A = torch.from_numpy(nd_arr).float()
    mask = torch.from_numpy(np.expand_dims(signal_mask, axis=0)[:, :, :]).float()
    print(A.shape)
    print(mask.shape)

    # Give batch dimension
    A = A.unsqueeze(0)
    mask = mask.unsqueeze(0)
    A = A.to(DEVICE)
    mask = mask.to(DEVICE)
    print(A.shape)
    print(mask.shape)

    # Return in same format as the dataloader
    return {"A" : A, "A_paths" : "null", "B_path" : "null", "mask" : mask }

def main(args):
    opt_Z = get_options(args.confif_Z, args.epoch_Z)
    opt_U = get_options(args.confif_U, args.epoch_U)
    opt_V = get_options(args.confif_V, args.epoch_V)
    opts = { "Z" : opt_Z, "U" : opt_Z, "opt_U" : opt_U }

    model_Z = get_model(opt_Z)
    model_U = get_model(opt_U)
    model_V = get_model(opt_V)
    models = { "Z" : model_Z, "U" : model_Z, "model_U" : model_U }

    f_in, f_out = get_h5_files(args.input_file, args.output_file, args.drop_3d, args.drop_projs)

    tot_evids = len(f_in["nd_packet_wire_projs"])
    for i_evid, evid in enumerate(f_in["nd_packet_wire_projs"].keys()):
        print(f"{i_evid} / {tot_evids}")
        for tpcset in f_in["nd_packet_wire_projs"][evid].keys():
            for rop in f_in["nd_packet_wire_projs"][evid][tpcset].keys():
                ret = get_plane_data(rop, args)
                signal_type = ret[0]
                plane_shape = ret[1]
                signalmask_max_tick_negative, signalmask_max_tick_positive = ret[2], ret[3]
                signalmask_max_ch = ret[4]
                opt = opts[signal_type]
                model = models[signal_type]

                nd_arr = make_nd_pixelmap(
                    f_in["nd_packet_wire_projs"][evid][tpcset][rop], plane_shape, False
                )
                signal_mask = make_signalmask(
                    nd_arr,
                    signalmask_max_tick_positive, signalmask_max_tick_negative, signalmask_max_ch
                )

                input_data = prep_model_input(opt, nd_arr, signal_mask)

                model.set_input(input_data, test=True)
                model.test()

                vis = model.get_current_visuals()
                pred = vis["fake_B"].cpu()
                mask = input_data["mask"].cpu()
                pred *= mask # XXX important, zeroing out predicted response outside of signal mask.
                pred = pred[0][0] # remove batch and adc channel dimensions
                pred /= opt.B_ch_scalefactors[0]
                pred = pred.numpy().astype(int)
                print(pred.shape)

                f_out.create_dataset("/pred_fd_resps/" + evid + tpcset + rop, data=pred)

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
    parser.add_argument("cache_dir", type=str, help="Dir to use as temporary disk space")

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

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_arguments()
    main(args)

