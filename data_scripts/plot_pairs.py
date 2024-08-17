import argparse, os

import numpy as np
import sparse
from matplotlib import pyplot as plt

def main(args):
    arrA = sparse.load_npz(args.nd_file).todense()
    arrA_ch = np.ma.masked_where((arrA[0] == 0) & (arrA[args.nd_channel] == 0), arrA[args.nd_channel])
    arrB = np.load(args.fd_file)[0]

    adc_abs_max = np.max(np.abs(arrB))
    vmin, vmax = -adc_abs_max, adc_abs_max

    fig, ax = plt.subplots(1, 2, figsize=(12, 8))

    ax[0].imshow(
        arrA_ch.T,
        cmap="viridis", aspect="auto", interpolation="none", origin="lower"
    )
    if args.show_signalmask:
        signalmask = np.ma.masked_where(arrA[-1] == 0, arrA[-1])
        ax[0].imshow(
            signalmask.T,
            cmap=plt.cm.gray, alpha=0.2, aspect="auto", interpolation="none", origin="lower"
        )
    if args.show_infillmask:
        infillmask = np.ma.masked_where(arrA[6] == 0, arrA[6])
        ax[0].imshow(
            infillmask.T,
            cmap=plt.cm.gray, alpha=0.2, aspect="auto", interpolation="none", origin="lower"
        )
    ax[0].set_title("ND")

    ax[1].imshow(
        arrB.T,
        cmap="seismic", aspect="auto", interpolation="none", origin="lower", vmin=vmin, vmax=vmax
    )
    if args.show_signalmask:
        ax[1].imshow(
            signalmask.T,
            cmap=plt.cm.gray, alpha=0.2, aspect="auto", interpolation="none", origin="lower"
        )
    ax[1].set_title("FD")

    plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("nd_file", type=str)
    parser.add_argument("fd_file", type=str)
    parser.add_argument("nd_channel", type=int)

    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument("--show_signalmask", action="store_true", help="overlay signal mask")
    group1.add_argument(
        "--show_infillmask", action="store_true",
        help="overlay infill reflection mask (assume channel 6)"
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main(parse_arguments())
