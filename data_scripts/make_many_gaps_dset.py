"""
Script to make new dataset with a lot of ND gaps from paired ND-FD dataset.
Only configured for collection currently.
"""
import os, argparse, random

import numpy as np
import sparse
from tqdm import tqdm


def main(args):
    for filename in tqdm(os.listdir(args.input_dir)):
        nd_arr = sparse.load_npz(os.path.join(args.input_dir, filename)).todense()

        nd_arr_nonzeros = np.nonzero(nd_arr[0])
        if nd_arr_nonzeros[0].size == 0:
            continue
        ch_minmax = (nd_arr_nonzeros[0].min(), nd_arr_nonzeros[0].max() - args.gap_chs)
        tick_minmax = (nd_arr_nonzeros[1].min(), nd_arr_nonzeros[1].max() - args.gap_ticks)

        num_ch_gaps = int(
            (ch_minmax[1] + args.gap_chs - ch_minmax[0]) /
            (args.avg_spacing_ch_gaps + 2 * args.gap_chs)
        )
        ch_gap_starts = []
        while len(ch_gap_starts) < num_ch_gaps:
            rand_ch = random.randint(*ch_minmax)
            if any(
                (
                    (ch_gap_start + 2 * args.gap_chs) > rand_ch and
                    (ch_gap_start - 2 * args.gap_chs) < rand_ch
                )
                for ch_gap_start in ch_gap_starts
            ):
                continue
            # Trying to avoid putting new gaps where there are already gaps
            # if (nd_arr[0][rand_ch:rand_ch + args.gap_chs, :].sum(1) == 0).sum() != 0:
            #     num_ch_gaps -=1
            #     continue
            ch_gap_starts.append(rand_ch)

        num_tick_gaps = int(
            (tick_minmax[1] + args.gap_ticks - tick_minmax[0]) /
            (args.avg_spacing_tick_gaps + 2 * args.gap_ticks)
        )
        tick_gap_starts = []
        while len(tick_gap_starts) < num_tick_gaps:
            rand_tick = random.randint(*tick_minmax)
            if any(
                (
                    (tick_gap_start + 2 * args.gap_ticks) > rand_tick and
                    (tick_gap_start - 2 * args.gap_ticks) < rand_tick
                )
                for tick_gap_start in tick_gap_starts
            ):
                continue
            # Trying to avoid putting new gaps where there are already gaps.
            # NOTE Not feasible here since there will be tick gaps between successive packets
            # if (nd_arr[0][:, rand_tick:rand_tick + args.gap_ticks].sum(1) == 0).sum() != 0:
            #     num_tick_gaps -=1
            #     continue
            tick_gap_starts.append(rand_tick)

        for ch_gap_start in ch_gap_starts:
            nd_arr[:6][ch_gap_start:ch_gap_start + args.gap_chs, :] = 0
        for tick_gap_start in tick_gap_starts:
            nd_arr[:6][:, tick_gap_start:tick_gap_start + args.gap_ticks] = 0

        nd_sarr = sparse.COO.from_numpy(nd_arr)
        sparse.save_npz(os.path.join(args.output_dir, filename), nd_sarr)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_dir")
    parser.add_argument("output_dir")

    parser.add_argument("--gap_chs", type=int, default=10)
    parser.add_argument("--gap_ticks", type=int, default=50)
    parser.add_argument("--avg_spacing_ch_gaps", type=int, default=50)
    parser.add_argument("--avg_spacing_tick_gaps", type=int, default=200)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

