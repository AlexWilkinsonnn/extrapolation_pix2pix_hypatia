import argparse, os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

plt.rc('font', family='serif')

def plot_images(realA, realB, fakeB, old_mode, ch, BtoA):
    if old_mode:
        adc_max = max([np.amax(realB), np.amax(fakeB)])
        adc_min = min([np.amin(realB), np.amin(fakeB)])
        adc_abs_max = np.abs(adc_max) if np.abs(adc_max) > np.abs(adc_min) else np.abs(adc_min)

    else:
        adc_max = max([np.amax(realB[0]), np.amax(fakeB[0])])
        adc_min = min([np.amin(realB[0]), np.amin(fakeB[0])])
        adc_abs_max = np.abs(adc_max) if np.abs(adc_max) > np.abs(adc_min) else np.abs(adc_min)

        if BtoA:
            realA = realA[ch]
            realB = realB[0]
            fakeB = fakeB[0].astype(int)
            realB = np.ma.masked_where(realB == 0, realB)
            fakeB = np.ma.masked_where(np.abs(fakeB) < 10, fakeB)
            
        else:
            realA = realA[ch]
            realB = realB[0]
            fakeB = fakeB[0]
            realA = np.ma.masked_where(realA == 0, realA)

    fig, ax = plt.subplots(1, 3)

    ax[0].imshow(realA.T, cmap='viridis', aspect='auto', interpolation='none')
    ax[0].set_title("realA")

    if realA.shape[0] == 480:
        cmap = 'viridis'
        vmin, vmax = adc_min, adc_max
    else:
        cmap = 'seismic'
        vmin, vmax = -adc_abs_max, adc_abs_max

    ax[1].imshow(fakeB.T, cmap=cmap, aspect='auto', interpolation='none', vmin=vmin, vmax=vmax)
    ax[1].set_title("fakeB")

    ax[2].imshow(realB.T, cmap=cmap, aspect='auto', interpolation='none', vmin=vmin, vmax=vmax)
    ax[2].set_title("realB")

    fig.tight_layout()
    plt.show()

def plot_channel_trace(realA, realB, fakeB, old_mode, BtoA):
    if not old_mode:
        realA = realA[0]
        realB = realB[0]
        fakeB = fakeB[0]
        
    ch = (0, 0)
    for idx, col in enumerate(realA):
        if np.abs(col).sum() > ch[1]:
            ch = (idx, np.abs(col).sum())
    ch = ch[0]

    tick_adc_true = realB[ch,:]
    tick_adc_fake = fakeB[ch,:]
    tick_energy_true = realA[ch,:]
    ticks = np.arange(1, realA.shape[1] + 1)
    
    fig, ax = plt.subplots()

    if BtoA:
        ax.hist(ticks, bins=len(ticks), weights=tick_adc_true, histtype='step', label="real_charge", linewidth=0.7)
        ax.hist(ticks, bins=len(ticks), weights=tick_adc_fake, histtype='step', label="fake_charge", linewidth=0.7)
        ax.set_ylabel("charge", fontsize=14)
    else:
        ax.hist(ticks, bins=len(ticks), weights=tick_adc_true, histtype='step', label="Ground Truth (ADC)", linewidth=0.8, color='#E69F00')
        ax.hist(ticks, bins=len(ticks), weights=tick_adc_fake, histtype='step', label="Output (ADC)", linewidth=0.8, color='#56B4E9')
        ax.set_ylabel("ADC", fontsize=14)        
    ax.set_xlabel("Tick", fontsize=14)
    ax.set_xlim(1, realA.shape[1] + 1)

    ax2 = ax.twinx()
    if BtoA:
        ax2.hist(ticks, bins=len(ticks), weights=tick_energy_true, histtype='step', label="real_adc", linewidth=0.7, color='g')
        ax2.set_ylabel("adc", fontsize=14)
        
    else:
        ax2.hist(ticks, bins=len(ticks), weights=tick_energy_true, histtype='step', label="Input (charge)", linewidth=0.8, color='#009E73')
        ax2.set_ylabel("charge", fontsize=14)

    ax_ylims = ax.axes.get_ylim()
    ax_yratio = ax_ylims[0] / ax_ylims[1]
    ax2_ylims = ax2.axes.get_ylim()
    ax2_yratio = ax2_ylims[0] / ax2_ylims[1]
    if ax_yratio < ax2_yratio:
        ax2.set_ylim(bottom = ax2_ylims[1]*ax_yratio)
    else:
        ax.set_ylim(bottom = ax_ylims[1]*ax2_yratio)
    
    plt.title("Channel {} in ROP".format(ch), fontsize=16)

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles += handles2
    labels += labels2
    new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
    plt.legend(handles=new_handles, labels=labels, prop={'size': 12})

    plt.show()

def main(input_dir, VALID_IMAGES, OLD_MODE, CHANNEL, BTOA, N):
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

            plot_images(realA, realB, fakeB, OLD_MODE, CHANNEL, BTOA)
            plot_channel_trace(realA, realB, fakeB, OLD_MODE, BTOA)

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

        plot_images(realA, realB, fakeB, OLD_MODE, CHANNEL, BTOA)
        plot_channel_trace(realA, realB, fakeB, OLD_MODE, BTOA)

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_dir")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--validation", dest='VALID_IMAGES', action='store_true')
    group.add_argument("--training", dest='TRAIN_IMAGES', action='store_true')

    parser.add_argument("--old_mode", dest='OLD_MODE', action='store_true', help="1 channel images were save as 2d arrays (no channel dimension)")
    parser.add_argument("--channel", type=int, dest='CHANNEL', default=0, help="channel of input image to plot")
    parser.add_argument("--BtoA", action='store_true', dest='BTOA')
    parser.add_argument("-n", type=int, default=3, dest='N')

    args = parser.parse_args()

    return (args.input_dir, args.VALID_IMAGES, args.OLD_MODE, args.CHANNEL, args.BTOA, args.N)

if __name__ == '__main__':
    arguments = parse_arguments()
    main(*arguments)
