import argparse, os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

plt.rc('font', family='serif')

def plot_images(realA, realB, fakeB, auto_crop):
    adc_max = max([np.amax(realB[0]), np.amax(fakeB[0])])
    adc_min = min([np.amin(realB[0]), np.amin(fakeB[0])])
    adc_abs_max = np.abs(adc_max) if np.abs(adc_max) > np.abs(adc_min) else np.abs(adc_min)

    realA = realA[0]
    realB = realB[0]
    fakeB = fakeB[0]
    realA = np.ma.masked_where(realA == 0, realA)

    if realA.shape[0] == 480:
        cmap = 'viridis'
        vmin, vmax = adc_min, adc_max
    else:
        cmap = 'seismic'
        vmin, vmax = -adc_abs_max, adc_abs_max

    if auto_crop:
        non_zeros = np.nonzero(realA)
        ch_min = non_zeros[0].min() - 10 if (non_zeros[0].min() - 10) > 0 else 0
        ch_max = non_zeros[0].max() + 11 if (non_zeros[0].max() + 11) < 480 else 480
        tick_min = non_zeros[1].min() - 50 if (non_zeros[1].min() - 50) > 0 else 0
        tick_max = non_zeros[1].max() + 51 if (non_zeros[1].max() + 51) < 4492 else 4492
        realA = realA[ch_min:ch_max, tick_min:tick_max]
        realB = realB[ch_min:ch_max, tick_min:tick_max]
        fakeB = fakeB[ch_min:ch_max, tick_min:tick_max]
        print("ch_min={}, cm_max={}, tick_min={}, tick_max={}".format(ch_min, ch_max, tick_min, tick_max))

    fig, ax = plt.subplots(1, 3)

    ax[0].imshow(realA.T, cmap='viridis', aspect='auto', interpolation='none', origin='lower')
    ax[0].set_title("realA")

    ax[1].imshow(fakeB.T, cmap=cmap, aspect='auto', interpolation='none', vmin=vmin, vmax=vmax, origin='lower')
    ax[1].set_title("fakeB")

    ax[2].imshow(realB.T, cmap=cmap, aspect='auto', interpolation='none', vmin=vmin, vmax=vmax, origin='lower')
    ax[2].set_title("realB")

    fig.tight_layout()
    plt.show()

def plot_channel_trace(realA, realB, fakeB, auto_crop):
    realA = realA[0]
    realB = realB[0]
    fakeB = fakeB[0]

    if auto_crop:
        non_zeros = np.nonzero(realA)
        ch_min = non_zeros[0].min() - 10 if (non_zeros[0].min() - 10) > 0 else 0
        ch_max = non_zeros[0].max() + 11 if (non_zeros[0].max() + 11) < 480 else 480
        tick_min = non_zeros[1].min() - 50 if (non_zeros[1].min() - 50) > 0 else 0
        tick_max = non_zeros[1].max() + 51 if (non_zeros[1].max() + 51) < 4492 else 4492
        realA = realA[ch_min:ch_max, tick_min:tick_max]
        realB = realB[ch_min:ch_max, tick_min:tick_max]
        fakeB = fakeB[ch_min:ch_max, tick_min:tick_max]
        
    ch = (0, 0)
    for idx, col in enumerate(realA):
        if np.abs(col).sum() > ch[1]:
            ch = (idx, np.abs(col).sum())
    ch = ch[0]

    tick_adc_true = realB[ch,:]
    tick_adc_fake = fakeB[ch,:]
    tick_adc_in = realA[ch,:]
    ticks = np.arange(1, realA.shape[1] + 1)
    
    fig, ax = plt.subplots()

    ax.hist(ticks, bins=len(ticks), weights=tick_adc_true, histtype='step', label="Ground Truth (FD ADC)", linewidth=0.8, color='#E69F00')
    ax.hist(ticks, bins=len(ticks), weights=tick_adc_fake, histtype='step', label="Output (FD ADC)", linewidth=0.8, color='#56B4E9')
    ax.hist(ticks, bins=len(ticks), weights=tick_adc_in, histtype='step', label="Input (FD ADC)", linewidth=0.8, color='#009E73')
    ax.set_ylabel("ADC", fontsize=14)        
    ax.set_xlabel("Tick", fontsize=14)
    ax.set_xlim(1, realA.shape[1] + 1)

    plt.title("Channel {} in ROP".format(ch), fontsize=16)

    handles, labels = ax.get_legend_handles_labels()
    new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
    plt.legend(handles=new_handles, labels=labels, prop={'size': 12})

    plt.show()

def main(input_dir, VALID_IMAGES, N, AUTO_CROP):
    if VALID_IMAGES:
        if N == 5:
            suffixes = ["valid0.npy", "valid1.npy", "valid2.npy","valid3.npy", "valid4.npy"]
        elif N == -5:
            suffixes = ["valid5.npy", "valid6.npy", "valid7.npy","valid8.npy", "valid9.npy"]
        else:
            suffixes = ["valid0.npy", "valid1.npy", "valid2.npy"]

        for suffix in suffixes:
            realA = np.load(os.path.join(input_dir, "realA_" + suffix))#[:,16:-16,:]
            realB = np.load(os.path.join(input_dir, "realB_" + suffix))#[:,16:-16,:]
            fakeB = np.load(os.path.join(input_dir, "fakeB_" + suffix))#[:,16:-16,:]
            
            """
            if realA.shape[2] == 512:
                realA = realA[:,16:-16,:]
                realB = realB[:,16:-16,:]
                fakeB = fakeB[:,16:-16,:]
            
            elif realA.shape[1] == 512:
                realA = realA[:,16:-16,58:-58]
                realB = realB[:,16:-16,58:-58]
                fakeB = fakeB[:,16:-16,58:-58]
            
            else:
                realA = realA[:,112:-112,58:-58]
                realB = realB[:,112:-112,58:-58]
                fakeB = fakeB[:,112:-112,58:-58]
            """

            plot_images(realA, realB, fakeB, AUTO_CROP)
            plot_channel_trace(realA, realB, fakeB, AUTO_CROP)

    else:
        realA = np.load(os.path.join(input_dir, "realA.npy"))
        realB = np.load(os.path.join(input_dir, "realB.npy"))
        fakeB = np.load(os.path.join(input_dir, "fakeB.npy"))

        if realA.shape[2] == 512:
            realA = realA[:,16:-16,:]
            realB = realB[:,16:-16,:]
            fakeB = fakeB[:,16:-16,:]
            
        elif realA.shape[1] == 512:
            realA = realA[:,16:-16,58:-58]
            realB = realB[:,16:-16,58:-58]
            fakeB = fakeB[:,16:-16,58:-58]
        
        else:
            realA = realA[:,112:-112,58:-58]
            realB = realB[:,112:-112,58:-58]
            fakeB = fakeB[:,112:-112,58:-58]

        plot_images(realA, realB, fakeB, AUTO_CROP)
        plot_channel_trace(realA, realB, fakeB, AUTO_CROP)

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_dir")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--validation", dest='VALID_IMAGES', action='store_true')
    group.add_argument("--training", dest='TRAIN_IMAGES', action='store_true')

    parser.add_argument("-n", type=int, default=3, dest='N')
    parser.add_argument("--auto_crop", action='store_true')

    args = parser.parse_args()

    return (args.input_dir, args.VALID_IMAGES, args.N, args.auto_crop)

if __name__ == '__main__':
    arguments = parse_arguments()
    main(*arguments)
