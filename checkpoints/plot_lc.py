import argparse, os

import numpy as np
from matplotlib import pyplot as plt
import yaml 

plt.rc('font', family='serif')

def main(input_dir, CH_LOSS, OLD_L1, ITER, VALID, DATASET, NO_PLOT):
    if DATASET == 0:
        train_size = 8000
        valid_freq = 4000
    elif DATASET == 1:
        train_size = 20000
        valid_freq = 10000
    elif DATASET == 2:
        train_size = 19000
        valid_freq = 9500
    elif DATASET == 3: # validUVZ
        train_size = 17000
        valid_freq = 8500
    elif DATASET == 4: # high res cropped
        train_size = 15000
        valid_freq = 7500
    elif DATASET == 5: # high res 8-8 cropped Z
        train_size = 17000
        valid_freq = 8500
    elif DATASET == 6: # high res 8-8 cropped V
        train_size = 1300
        valid_freq = 6500

    with open(os.path.join(input_dir, "loss.txt"), 'r') as f:
        lines = f.readlines()

        G_GAN_epoch, G_GAN = [], []
        G_pix_epoch, G_pix = [], []
        G_pix_valid = []

        if CH_LOSS:
            G_channel_epoch, G_channel = [], []
            G_channel_valid = []

        if ITER:
            current_iter = ITER
            for line in lines:
                if int(line.split("total_iters=")[1].split(',')[0]) >= current_iter:
                    current_iter += ITER
                    G_GAN.append(np.mean(G_GAN_epoch))
                    G_GAN_epoch.clear()
                    G_pix.append(np.mean(G_pix_epoch))
                    G_pix_epoch.clear()
                    if CH_LOSS:
                        G_channel.append(np.mean(G_channel_epoch))
                        G_channel_epoch.clear()

                if line.startswith('VALID:'):
                    if len(G_pix_valid) == 0:
                        iter1 = int(line.split("total_iters=")[1].split(',')[0])
                    if len(G_pix_valid) == 1:
                        iter2 = int(line.split("total_iters=")[1].split(',')[0])
                    G_pix_valid.append(float(line.split("G_pix=")[1].split(',')[0]))
                    if CH_LOSS:
                        G_channel_valid.append(float(line.split("G_channel=")[1].split(',')[0]))
                    
                else:
                    G_GAN_epoch.append(float(line.split("G_GAN=")[1].split(',')[0]))
                    if OLD_L1:
                        G_pix_epoch.append(float(line.split("G_L1=")[1].split(',')[0]))
                    else:
                        G_pix_epoch.append(float(line.split("G_pix=")[1].split(',')[0]))
                        
                    if CH_LOSS:
                        G_channel_epoch.append(float(line.split("G_channel=")[1].split(',')[0]))

        else:
            current_epoch = 1
            for line in lines:
                if int(line.split("epoch=")[1].split(',')[0]) != current_epoch:
                    current_epoch += 1
                    G_GAN.append(np.mean(G_GAN_epoch))
                    G_GAN_epoch.clear()
                    G_pix.append(np.mean(G_pix_epoch))
                    G_pix_epoch.clear()
                    if CH_LOSS:
                        G_channel.append(np.mean(G_channel_epoch))
                        G_channel_epoch.clear()

                if line.startswith('VALID'):
                    G_pix_valid.append(float(line.split("G_pix=")[1].split(',')[0]))
                    if CH_LOSS:
                        G_channel_valid.append(float(line.split("G_channel=")[1].split(',')[0]))
                    
                else:
                    G_GAN_epoch.append(float(line.split("G_GAN=")[1].split(',')[0]))
                    if OLD_L1:
                        G_pix_epoch.append(float(line.split("G_L1=")[1].split(',')[0]))
                    else:
                        G_pix_epoch.append(float(line.split("G_pix=")[1].split(',')[0]))
                        
                    if CH_LOSS:
                        G_channel_epoch.append(float(line.split("G_channel=")[1].split(',')[0]))

    if ITER:
        x = np.arange(ITER, (len(G_GAN) + 1)*ITER, ITER)
    else:
        x = np.arange(1, len(G_GAN) + 1)

    valid_iter_interval = iter2 - iter1
    valid_x = np.arange(valid_iter_interval, (len(G_pix_valid) + 1)*valid_iter_interval, valid_iter_interval)
    G_pix_valid_min, G_pix_valid_min_idx = min((val, idx) for (idx, val) in enumerate(G_pix_valid))
    G_channel_valid_min, G_channel_valid_min_idx = min((val, idx) for (idx, val) in enumerate(G_channel_valid))
    print("G_pix_valid={}".format(G_pix_valid))
    print("min(G_pix_valid)={} at {} iter -> {} epoch assuming {} iter to an epoch + valid every {} iter".format(
        G_pix_valid_min, (G_pix_valid_min_idx + 1)*valid_freq, (G_pix_valid_min_idx + 1)*valid_freq/train_size, train_size, valid_freq))
    print("G_channel_valid={}".format(G_channel_valid))
    print("min(G_channel_valid)={} at {} iter -> {} epoch assuming {} iter to an epoch + valid every {} iter".format(
        G_channel_valid_min, (G_channel_valid_min_idx + 1)*valid_freq, (G_channel_valid_min_idx + 1)*valid_freq/train_size, train_size, valid_freq))
    
    if os.path.isfile(os.path.join(input_dir, 'best_metrics.yaml')):
        with open(os.path.join(input_dir, 'best_metrics.yaml')) as f:
            best_metrics = yaml.load(f, Loader=yaml.FullLoader)
        print("Best metrics (models saved):")
        for key, value in best_metrics.items():
            if key.endswith("itr"):
                continue
            if type(value) is tuple:
                print("{: >25}: {: >21}, {: >21} at {: >20} itr {: >6} epochs".format(key, value[0], value[1], best_metrics[key + '_itr'], str(round(best_metrics[key + '_itr']/train_size, 2))))
            else:
                print("{: >25}: {: >44} at {: >20} itr {: >6} epochs".format(key, value, best_metrics[key + '_itr'], str(round(best_metrics[key + '_itr']/train_size, 2))))
            
    if NO_PLOT:
        return

    fig, ax = plt.subplots(figsize=(16,8))

    lns1 = ax.plot(x, G_GAN, label="G_GAN", color='b', alpha=0.5)
    ax.set_ylabel("G_GAN", fontsize=12)
    x_label = "epoch" if not ITER else "iter"
    ax.set_xlabel(x_label, fontsize=12)
#    ax.set_ylim(bottom=(min(G_GAN) - 0.1*min(G_GAN)), top=(max(G_GAN) + 0.1*max(G_GAN)))

    ax2 = ax.twinx()
    lns2 = ax2.plot(x, G_pix, label="G_L1", color='r', alpha=0.5)
    ax2.set_ylabel("G_pix", fontsize=12)
#    ax2.set_ylim(bottom=(min(G_pix) - 0.1*min(G_pix)), top=(max(G_pix) + 0.1*max(G_pix)))
    if VALID:
        ax3 = ax.twinx()
        lns3 = ax3.plot(valid_x, G_pix_valid, label="G_L1_valid", color='m')
        ax3.set_ylabel("G_pix_valid", fontsize=12)
#        ax3.set_ylim(bottom=(min(G_pix_valid) - 0.1*min(G_pix_valid)), top=(max(G_pix_valid) + 0.1*max(G_pix_valid)))
        ax3.spines['right'].set_position(('outward', 70))
        ax3.grid()

    if CH_LOSS:
        ax4 = ax.twinx()
        lns4 = ax4.plot(x, G_channel, label="G_channel", color='g', alpha=0.5)
        ax4.set_ylabel("G_channel", fontsize=14)
        ax4.spines['right'].set_position(('outward', 140))
        if VALID:
            ax5 = ax.twinx()
            lns5 = ax5.plot(valid_x, G_channel_valid, label="G_channel_valid", color='y')
            ax5.set_ylabel("G_channel_valid", fontsize=12)
            ax5.spines['right'].set_position(('outward', 210))
                    
    lns = lns1 + lns2
    if CH_LOSS:
        lns += lns4
    if VALID:
        lns += lns3
        lns += lns5
    labs = [ l.get_label() for l in lns ]
    ax.legend(lns, labs, loc=0)

    fig.tight_layout()
    plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_dir")
    
    # Maybe want an option to plot loss by iteration over the epoch loss?
    parser.add_argument("--ch_loss", action='store_true', dest='CH_LOSS')
    parser.add_argument("--old_L1", action='store_true', dest='OLD_L1')
    parser.add_argument("--validation", action='store_true', dest='VALID', help="must be using iter")
    parser.add_argument("--iter", type=int, dest='ITER', default=0,
        help="get losses every n iterations instead of every epoch")
    parser.add_argument("--dataset", type=int, default=0, help="Which iteration of the dataset is being used")
    parser.add_argument("--no_plot", action='store_true')

    args = parser.parse_args()

    return (args.input_dir, args.CH_LOSS, args.OLD_L1, args.ITER, args.VALID, args.dataset, args.no_plot)

if __name__ == '__main__':
    arguments = parse_arguments()
    main(*arguments)
