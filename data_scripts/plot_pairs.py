import argparse, os

import numpy as np
import sparse
from matplotlib import pyplot as plt

def main(args):
    arrA = sparse.load_npz(args.nd_file).todense()
    arrA = np.ma.masked_where(arrA[0] == 0, arrA[args.nd_channel])
    arrB = np.load(args.fd_file)[0]

    adc_abs_max = np.max(np.abs(arrB))
    vmin, vmax = -adc_abs_max, adc_abs_max

    fig, ax = plt.subplots(1, 2, figsize=(12, 8))

    ax[0].imshow(
        arrA.T,
        cmap="viridis", aspect="auto", interpolation="none", origin="lower"
    )
    ax[0].set_title("ND")

    ax[1].imshow(
        arrB.T,
        cmap="seismic", aspect="auto", interpolation="none", origin="lower", vmin=vmin, vmax=vmax
    )
    ax[1].set_title("FD")

    plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("nd_file", type=str)
    parser.add_argument("fd_file", type=str)
    parser.add_argument("nd_channel", type=int)
    parser.add_argument("signal_type", type=str, help="collection|induction")

    args = parser.parse_args()

    if args.signal_type not in ["collection", "induction"]:
        raise argparse.ArgumentError(f"signal_type {args.signal_type} not valid")

    return args

if __name__ == "__main__":
    main(parse_arguments())
