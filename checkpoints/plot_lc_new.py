import argparse, os
from itertools import cycle

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import yaml

plt.rc("font", family="serif")

def main(args):
    with open(os.path.join(args.input_dir, "loss.txt"), "r") as f:
        losses = [ loss_line.rstrip() for loss_line in f ]

    with open(os.path.join(args.input_dir, "config.yaml"), "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    lambda_pix, lambda_ch = conf["lambda_pix"], conf["lambda_channel"]
    
    train_epochs = []
    train_losses_pix, train_losses_ch, train_losses_GAN, train_losses_total = [], [], [], []
    train_losses_D_real, train_losses_D_fake = [], []
    val_epochs, val_losses_pix, val_losses_ch = [], [], []

    itrs_per_epoch = int(losses.pop(0).split(": ")[1])

    for i_line, line in enumerate(losses):
        if line.startswith("Epoch: "):
            epoch = int(line.split("Epoch: ")[1].split(" ")[0])
            itr = int(line.split("Iter: ")[1].split(" ")[0]) # assuming Iter: is before Total Iter:
            train_epochs.append(epoch + itr / itrs_per_epoch)
            loss_line = line.split(" -- ")[1]
            train_losses_pix.append(
                float(loss_line.split("G_pix=")[1].split(" ")[0]) * lambda_pix
            )
            train_losses_ch.append(
                float(loss_line.split("G_channel=")[1].split(" ")[0]) * lambda_ch
            )
            train_losses_GAN.append(float(loss_line.split("G_GAN=")[1].split(" ")[0]))
            train_losses_total.append(
                train_losses_pix[-1] + train_losses_ch[-1] + train_losses_GAN[-1]
            )
            train_losses_D_real.append(float(loss_line.split("D_real=")[1].split(" ")[0]))
            train_losses_D_fake.append(float(loss_line.split("D_fake=")[1].split(" ")[0]))

        elif line.startswith("VALID: "):
            epoch = int(line.split("Epoch: ")[1].split(" ")[0])
            val_epochs.append(epoch) # assuming valid_freq = "epoch"
            loss_line = line.split(" -- ")[1]
            val_losses_pix.append(
                float(loss_line.split("G_pix=")[1].split(" ")[0]) * lambda_pix
            )
            val_losses_ch.append(
                float(loss_line.split("G_channel=")[1].split(" ")[0]) * lambda_ch
            )

    fig, ax = plt.subplots()

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = cycle(prop_cycle.by_key()['color'])

    c_pix = next(colors)
    ax.plot(train_epochs, train_losses_pix, label="train G_pix", alpha=0.5, c=c_pix)
    c_ch = next(colors)
    ax.plot(train_epochs, train_losses_ch, label="train G_channel", alpha=0.5, c=c_ch)
    c_GAN = next(colors)
    ax.plot(train_epochs, train_losses_GAN, label="train G_GAN", alpha=0.5, c=c_GAN)
    if args.show_D_losses:
        c_D = next(colors)
        ax.plot(
            train_epochs, train_losses_D_real,
            label="train D_real", alpha=1.0, linestyle="--", c=c_D
        )
        ax.plot(
            train_epochs, train_losses_D_fake,
            label="train D_fake", alpha=1.0, c=c_D, linestyle=":"
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    c_val_pix = next(colors)
    ax.plot(val_epochs, val_losses_pix, label="valid G_pix", c=c_val_pix)
    c_val_ch = next(colors)
    ax.plot(val_epochs, val_losses_ch, label="valid G_channel", c=c_val_ch)
    ax.set_ylabel("Valid losses")

    plt.legend(ncol=2)

    plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_dir", help="dir with loss.txt file")

    parser.add_argument("--show_D_losses", action="store_true")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main(parse_arguments())
