"""
Script to make training dataset from paired ndfd hdf5 files. The ND wire projection image will
contained a channel with an infilled flag and a reflection mask flag. A signal mask is made by
smearing the ND input image will be made and added to the last channel of the FD array.
"""
import os, argparse
from collections import defaultdict
from functools import partialmethod

import numpy as np
import sparse, h5py
from tqdm import tqdm

def main(args):
    if args.batch_mode:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    out_dir_A = os.path.join(args.output_dir, "allA")
    if not os.path.exists(out_dir_A):
        raise ValueError(f"Output path {out_dir_A} does not exist")
    out_dir_B = os.path.join(args.output_dir, "allB")
    if not os.path.exists(out_dir_B):
        raise ValueError(f"Output path {out_dir_B} does not exist")

    f = h5py.File(os.path.join(args.input_file))

    tot_evids = len(f["nd_packet_wire_projs"])
    cntr = args.start_idx
    
    for i_evid, evid in enumerate(f["nd_packet_wire_projs"].keys()):
        print(f"\n----\n{i_evid} / {tot_evids}")
        for tpcset in f["nd_packet_wire_projs"][evid].keys():
            for rop in f["nd_packet_wire_projs"][evid][tpcset].keys():
                if (
                    args.signal_type == "Z" and rop != "2" and rop != "3" or
                    args.signal_type == "V" and rop != "1" or
                    args.signal_type == "U" and rop != "0"
                ):
                    continue

                fd_arr = make_fd_pixelmap(f["fd_resp"][evid][tpcset][rop][:])
                nd_arr = make_nd_pixelmap(
                    f["nd_packet_wire_projs"][evid][tpcset][rop],
                    fd_arr.shape[1:],
                    args.reflection_mask
                )
                if np.sum(nd_arr[0]) < args.min_adc:
                    continue
                signal_mask = make_signalmask(
                    nd_arr,
                    args.signalmask_max_tick_positive, args.signalmask_max_tick_negative,
                    args.signalmask_max_ch
                )
                nd_arr = np.concatenate([nd_arr, np.expand_dims(signal_mask, axis=0)], 0)

                nd_sarr = sparse.COO.from_numpy(nd_arr)
                sparse.save_npz(os.path.join(out_dir_A, f"{cntr}nd.npz"), nd_sarr)
                np.save(os.path.join(out_dir_B, f"{cntr}fd.npy"), fd_arr)

                cntr += 1

def make_fd_pixelmap(data):
    return np.expand_dims(data, axis=0).astype(float)

def make_signalmask(nd_pixelmap, max_tick_shift_positive, max_tick_shift_negative, max_ch_shift):
    nd_adcs = nd_pixelmap[0]
    mask = np.copy(nd_adcs)

    for _ in range(1, max_tick_shift_positive + 1):
        mask[:, 1:] += mask[:, :-1]
    for _ in range(1, max_tick_shift_negative + 1):
        mask[:, :-1] += mask[:, 1:]
    
    for _ in range(1, max_ch_shift + 1):
        mask[1:, :] += mask[:-1, :]
        mask[:-1, :] += mask[1:, :]

    return mask.astype(bool).astype(float)

def make_nd_pixelmap(data, plane_shape, reflection_mask):
    arr = np.zeros((7 if reflection_mask else 6, *plane_shape), dtype=float)
    adc_weighted_avg_numerators = {
        "nd_drift_dist" : defaultdict(float),
        "fd_drift_dist" : defaultdict(float),
        "wire_dist" : defaultdict(float),
        "infilled" : defaultdict(float),
    }

    for i in tqdm(range(len(data))):
        ch = data["local_ch"][i]
        tick = data["tick"][i]
        chtick = (ch, tick)
        adc = float(data["adc"][i])

        arr[0, ch, tick] += adc

        if arr[0, ch, tick]:
            nd_drift_dist = float(data["nd_drift_dist"][i])
            # Some nd drift distance are slightly -ve due to drift distance calculation weirdness
            # Treat these as zero
            if nd_drift_dist > 0.0:
                adc_weighted_avg_numerators["nd_drift_dist"][chtick] += adc * nd_drift_dist
                arr[1, ch, tick] = (
                    adc_weighted_avg_numerators["nd_drift_dist"][chtick] / arr[0, ch, tick]
                )

            fd_drift_dist = float(data["fd_drift_dist"][i])
            adc_weighted_avg_numerators["fd_drift_dist"][chtick] += adc * fd_drift_dist
            arr[2, ch, tick] = (
                adc_weighted_avg_numerators["fd_drift_dist"][chtick] / arr[0, ch, tick]
            )

            wire_dist = float(data["wire_dist"][i])
            adc_weighted_avg_numerators["wire_dist"][chtick] += adc * wire_dist
            arr[4, ch, tick] = (
                adc_weighted_avg_numerators["wire_dist"][chtick] / arr[0, ch, tick]
            )

            infilled = float(data["infilled"][i])
            adc_weighted_avg_numerators["infilled"][chtick] += adc * infilled
            arr[5, ch, tick] = (
                adc_weighted_avg_numerators["infilled"][chtick] / arr[0, ch, tick]
            )

        if reflection_mask:
            # Projection is from the reflection mask, dont want this included in stack
            if data["infilled"][i] == 2:
                arr[6, ch, tick] = 1.0
            else:
                arr[3, ch, tick] += 1 # number of stacked packets in (wire, tick)
        else:
            arr[3, ch, tick] += 1 # number of stacked packets in (wire, tick)

    arr[1] = np.sqrt(arr[1])
    arr[2] = np.sqrt(arr[2])

    return arr

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file", type=str, help="ndfd h5 file")
    parser.add_argument(
        "output_dir", type=str,
        help="output dir where 'allA' and allB' directories will be made a filled"
    )
    parser.add_argument("signal_type", type=str, help="Z|U|V")
    
    parser.add_argument(
        "--min_adc", type=int, default=0,
        help="minimum ND adc on a readout plane to make a pair"
    )
    parser.add_argument(
        "--start_idx", type=int, default=0,
        help="number to start naming output pairs from"
    )
    parser.add_argument(
        "--signalmask_max_tick_positive", type=int, default=None,
        help=(
            "Max ND packet smearing in positive tick direction to make signal mask."
            "Default 30 for Z, 50 for U|V"
        )
    )
    parser.add_argument(
        "--signalmask_max_tick_negative", type=int, default=None,
        help=(
            "Max ND packet smearing in negative tick direction to make signal mask."
            "Default 30 for Z, 50 for U|V"
        )
    )
    parser.add_argument(
        "--signalmask_max_ch", type=int, default=None,
        help=(
            "Max ND packet smearing in ch direction to make signal mask."
            "Default 4 for Z|U|V"
        )
    )
    parser.add_argument("--batch_mode", action="store_true")
    parser.add_argument("--reflection_mask", action="store_true")

    args = parser.parse_args()

    if args.signal_type not in ["Z", "U", "V"]:
        raise argparse.ArgumentError(f"signal_type {args.signal_type} is not valid")

    if args.signalmask_max_tick_positive is None:
        args.signalmask_max_tick_positive = 30 if args.signal_type == "Z" else 50
    if args.signalmask_max_tick_negative is None:
        args.signalmask_max_tick_negative = 30 if args.signal_type == "Z" else 50
    if args.signalmask_max_ch is None:
        args.signalmask_max_tick = 4

    return args

if __name__ == "__main__":
    main(parse_arguments())
