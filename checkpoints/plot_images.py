import argparse, os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

plt.rc('font', family='serif')

def plot_images(realA, realB, fakeB, auto_crop, pdf, induction, jet=False):
    if len(realA.shape) == 3:
        realA = realA[0]
        realB = realB[0]
        fakeB = fakeB[0]

    adc_max = max([np.amax(realB), np.amax(fakeB)])
    adc_min = min([np.amin(realB), np.amin(fakeB)])
    adc_abs_max = np.abs(adc_max) if np.abs(adc_max) > np.abs(adc_min) else np.abs(adc_min)

    # if realA.shape[0] == 800:
    if induction:
        cmap = 'seismic'
        vmin, vmax = -adc_abs_max, adc_abs_max
    else: 
        cmap = 'viridis' if not jet else "jet"
        vmin, vmax = adc_min, adc_max

    realA = np.ma.masked_where(realA == 0, realA)

    if auto_crop:
        non_zeros = np.nonzero(realA)
        ch_min = non_zeros[0].min() - 10 if (non_zeros[0].min() - 10) > 0 else 0
        ch_max = non_zeros[0].max() + 11 if (non_zeros[0].max() + 11) < realA.shape[0] else realA.shape[0]
        tick_min = non_zeros[1].min() - 50 if (non_zeros[1].min() - 50) > 0 else 0
        tick_max = non_zeros[1].max() + 51 if (non_zeros[1].max() + 51) < 4492 else 4492
        realA = realA[ch_min:ch_max, tick_min:tick_max]
        realB = realB[ch_min:ch_max, tick_min:tick_max]
        fakeB = fakeB[ch_min:ch_max, tick_min:tick_max]
        print("ch_min={}, ch_max={}, tick_min={}, tick_max={}".format(ch_min, ch_max, tick_min, tick_max))

    if type(pdf) == PdfPages:
        fig, ax = plt.subplots(1, 3, figsize=(12,6))
    else:
        fig, ax = plt.subplots(1, 3)

    ax[0].imshow(realA.T, cmap='viridis', aspect='auto', interpolation='none', origin='lower')
    ax[0].set_title("realA")

    ax[1].imshow(fakeB.T, cmap=cmap, aspect='auto', interpolation='none', vmin=vmin, vmax=vmax, origin='lower')
    ax[1].set_title("fakeB")

    ax[2].imshow(realB.T, cmap=cmap, aspect='auto', interpolation='none', vmin=vmin, vmax=vmax, origin='lower')
    ax[2].set_title("realB")

    fig.tight_layout()
    if type(pdf) == PdfPages:
        pdf.savefig(bbox_inches='tight')
    else:
        plt.show()

def plot_channel_trace(realA, realB, fakeB, auto_crop, pdf, downres):
    if len(realA.shape) == 3:
        realA = realA[0]
        realB = realB[0]
        fakeB = fakeB[0]

    if auto_crop:
        non_zeros = np.nonzero(realA)
        ch_min = non_zeros[0].min() - 10 if (non_zeros[0].min() - 10) > 0 else 0
        ch_max = non_zeros[0].max() + 11 if (non_zeros[0].max() + 11) < realA.shape[0] else realA.shape[0]
        tick_min = non_zeros[1].min() - 50 if (non_zeros[1].min() - 50) > 0 else 0
        tick_max = non_zeros[1].max() + 51 if (non_zeros[1].max() + 51) < 4492 else 4492
        realA = realA[ch_min:ch_max, tick_min:tick_max]
        realB = realB[ch_min:ch_max, tick_min:tick_max]
        fakeB = fakeB[ch_min:ch_max, tick_min:tick_max]

    if downres:
        if downres == '4,10':
            ch_factor, tick_factor = 4, 10
        elif downres == '8,8':
            ch_factor, tick_factor = 8, 8
        else:
            raise NotImplementedError()
            
        realA_downres = np.zeros((int(realA.shape[0]/ch_factor), int(realA.shape[1]/tick_factor)))
        for ch, ch_vec in enumerate(realA):
            for tick, adc in enumerate(ch_vec):
                realA_downres[int(ch/ch_factor), int(tick/tick_factor)] += adc
        realA = realA_downres
        
    ch = (0, 0)
    for idx, col in enumerate(realA):
        if np.abs(col).sum() > ch[1]:
            ch = (idx, np.abs(col).sum())
    ch = ch[0]

    tick_adc_true = realB[ch,:]
    tick_adc_fake = fakeB[ch,:]
    tick_adc_in = realA[ch,:]
    ticks = np.arange(1, realA.shape[1] + 1)

    if type(pdf) == PdfPages:
        fig, ax = plt.subplots(figsize=(12,4))
    else:
        fig, ax = plt.subplots()    

    ax.hist(ticks, bins=len(ticks), weights=tick_adc_true, histtype='step', label="Ground Truth (FD ADC)", linewidth=0.8, color='#E69F00')
    ax.hist(ticks, bins=len(ticks), weights=tick_adc_fake, histtype='step', label="Output (FD ADC)", linewidth=0.8, color='#56B4E9')
    ax.hist(ticks, bins=len(ticks), weights=tick_adc_in, histtype='step', label="Input (ND ADC)", linewidth=0.8, color='#009E73')
    ax.set_ylabel("ADC", fontsize=14)        
    ax.set_xlabel("Tick", fontsize=14)
    ax.set_xlim(1, realA.shape[1] + 1)

    plt.title("Channel {} in ROP".format(ch), fontsize=16)

    handles, labels = ax.get_legend_handles_labels()
    new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
    plt.legend(handles=new_handles, labels=labels, prop={'size': 12})

    if type(pdf) == PdfPages:
        pdf.savefig(bbox_inches='tight')
    else:
        plt.show()

def main(input_dir, VALID_IMAGES, N, AUTO_CROP, PDF, DOWNRES, JET):
    if PDF:
        pdf = PdfPages('out.pdf')
    else:
        pdf = False

    induction = False if 'Z' in input_dir else True
    # induction = True

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

            plot_images(realA, realB, fakeB, AUTO_CROP, pdf, induction, jet=JET)
            plot_channel_trace(realA, realB, fakeB, AUTO_CROP, pdf, DOWNRES)

        if PDF:
            pdf.close()

    else:
        realA = np.load(os.path.join(input_dir, "realA.npy"))
        realB = np.load(os.path.join(input_dir, "realB.npy"))
        fakeB = np.load(os.path.join(input_dir, "fakeB.npy"))
        
        """
        if realA.shape[1] == 512:
            realA = realA[:,16:-16,58:-58]
            realB = realB[:,16:-16,58:-58]
            fakeB = fakeB[:,16:-16,58:-58]
        
        else:
            realA = realA[:,112:-112,58:-58]
            realB = realB[:,112:-112,58:-58]
            fakeB = fakeB[:,112:-112,58:-58]
        """

        plot_images(realA, realB, fakeB, AUTO_CROP, pdf, induction, jet=JET)
        plot_channel_trace(realA, realB, fakeB, AUTO_CROP, pdf, DOWNRES)

        if PDF:
            pdf.close()
    

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_dir")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--validation", dest='VALID_IMAGES', action='store_true')
    group.add_argument("--training", dest='TRAIN_IMAGES', action='store_true')

    parser.add_argument("-n", type=int, default=3, dest='N')
    parser.add_argument("--auto_crop", action='store_true')
    parser.add_argument("--pdf", action='store_true')
    parser.add_argument("--downres", type=str, default='')
    parser.add_argument("--jet", action="store_true")

    args = parser.parse_args()

    if args.auto_crop and args.downres:
        raise NotImplementedError("Not sure how to implement auto_crop for the downres translation")

    return (
        args.input_dir, args.VALID_IMAGES, args.N, args.auto_crop, args.pdf, args.downres, args.jet
    )

if __name__ == '__main__':
    arguments = parse_arguments()
    main(*arguments)
