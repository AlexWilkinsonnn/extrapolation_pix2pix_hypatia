"""
Script to make training dataset from paired ndfd hdf5 files. The ND wire projection image will
contained a channel with an infilled flag
"""
import os, argparse
from collections import defaultdict

import numpy as np
import sparse, h5py
from tqdm import tqdm

def main(args):
    out_dir_A = os.path.join(args.output_dir, "allA")
    if not os.path.exists(out_dir_A):
        os.makedirs(out_dir_A)
    out_dir_B = os.path.join(args.output_dir, "allB")
    if not os.path.exists(out_dir_B):
        os.makedirs(out_dir_B)

    cntr = 0

    for fname in tqdm(os.listdir(args.input_dir)):
        try:
            f = h5py.File(os.path.join(args.input_dir, fname))
        except Exception as e:
            print(f"Opening hdf5 file {fname} failed with:\n{e}")
            continue
        
        for evid in f["nd_packet_wire_projs"].keys():
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
                        f["nd_packet_wire_projs"][evid][tpcset][rop], fd_arr.shape[1:]
                    )

                    if np.sum(nd_arr[0]) < args.min_adc:
                        continue

                    nd_sarr = sparse.COO.from_numpy(nd_arr)
                    sparse.save_npz(os.path.join(out_dir_A, f"{cntr}nd.npz"), nd_sarr)
                    np.save(os.path.join(out_dir_B, f"{cntr}fd.npy"), fd_arr)

                    cntr += 1

def make_fd_pixelmap(data):
    return np.expand_dims(data, axis=0).astype(np.int16)

def make_nd_pixelmap(data, plane_shape):
    arr = np.zeros((6, *plane_shape), dtype=float)
    adc_weighted_avg_numerators = {
        "nd_drift_dist" : defaultdict(float),
        "fd_drift_dist" : defaultdict(float),
        "wire_dist" : defaultdict(float),
        "infilled" : defaultdict(float)
    }

    for i in range(len(data)):
        ch = data["local_ch"][i]
        tick = data["tick"][i]
        chtick = (ch, tick)
        adc = float(data["adc"][i])

        arr[0, ch, tick] += adc

        if arr[0, ch, tick]:
            nd_drift_dist = float(data["nd_drift_dist"][i])
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

        arr[3, ch, tick] += 1

    arr[1] = np.sqrt(arr[1])
    arr[2] = np.sqrt(arr[2])

    return arr

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_dir", type=str, help="dir with ndfd h5 files")
    parser.add_argument(
        "output_dir", type=str,
        help="output dir where 'allA' and allB' directories will be made a filled"
    )
    parser.add_argument("signal_type", type=str, help="Z|U|V")
    
    parser.add_argument(
        "--min_adc", type=int, default=0,
        help="minimum ND adc on a readout plane to make a pair"
    )

    args = parser.parse_args()

    if args.signal_type not in ["Z", "U", "V"]:
        raise argparse.ArgumentError(f"signal_type {args.signal_type} is not valid")

    return args


if __name__ == "__main__":
    main(parse_arguments())
