import os, argparse

import numpy as np
from matplotlib import pyplot as plt
import sparse

"""
Script to replace signal mask made from ND packets with one made from smearing the FD signal.
Doing this to ensure that gaps that are to be infilled are included in the signal mask.
The mask looks quite large but need to account for induction signals cancelling out.
"""

def main(ND_DIR, FD_DIR):
    for ientry, entry in enumerate(os.scandir(ND_DIR)):
        print(ientry, end='\r')

        arr_nd = sparse.load_npz(entry.path).todense()
        arr_fd = np.load(os.path.join(FD_DIR, entry.name.split('nd.npz')[0] + 'fd.npy'))

        new_mask = np.abs(arr_fd[0]) > 20

        max_tick_shift = 10 if arr_fd.shape[1] == 480 else 15
        max_ch_shift = 3

        for tick_shift in range(1, max_tick_shift + 1):
            new_mask[:, tick_shift:] += new_mask[:, :-tick_shift]
            new_mask[:, :-tick_shift] += new_mask[:, tick_shift:]

        for ch_shift in range(1, max_ch_shift + 1):
            new_mask[ch_shift:, :] += new_mask[:-ch_shift, :]
            new_mask[:-ch_shift, :] += new_mask[ch_shift:, :]

        arr_nd[-1] = new_mask.astype(float)

        sarr_nd = sparse.COO.from_numpy(arr_nd)
        sparse.save_npz(entry.path, sarr_nd)

        # mask_comparison = arr_nd[-1] + (new_mask * 2)
        # plt.imshow(np.ma.masked_where(mask_comparison == 0, mask_comparison).T, aspect='auto', interpolation='none', cmap='jet')
        # plt.show()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("nd_dir", type=str,
                        help="dir of ND data to add infill mask to")
    parser.add_argument("fd_dir", type=str,
                        help="dir of FD data to make mask from")

    args = parser.parse_args()

    return (args.nd_dir, args.fd_dir)

if __name__ == '__main__':
    args = parse_args()
    main(*args)
