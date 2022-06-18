import os, argparse, sys
from collections import namedtuple

import torch

import yaml
import numpy as np

from model import *
from dataset import *

def main(EXPERIMENT_DIR, EPOCH, OUTPUT_NAME):
    opt = load_config(EXPERIMENT_DIR, EPOCH)

    dataset = CustomDatasetDataLoader(opt, valid=True).load_data()
    dataset_iterator = iter(dataset)
    print("Data loaded from {}".format(opt.dataroot))

    model = Pix2pix(opt)
    print("model [%s] was created" % type(model).__name__)
    model.setup(opt)
    model.eval()

    data = next(dataset_iterator)

    A = data['A']
    G = model.get_netG()



def load_config(experiment_dir: str, epoch: str) -> namedtuple:
    with open(os.path.join(experiment_dir, 'config.yaml')) as f:
        options = yaml.load(f, Loader=yaml.FullLoader)

    # If data is not on the current node, grab it from the share disk.
    if not os.path.exists(options['dataroot']):
        options['dataroot'] = options['dataroot_shared_disk']

    # Settings for JIT
    options['gpu_ids'] = [-1] # Tells init function not to wrap model with DataParallel
    options['num_threads'] = 1
    options['phase'] = 'test'
    options['isTrain'] = False

    options['epoch'] = epoch

    print("Using configuration:")
    for key, value in options.items():
        print("{}={}".format(key, value))

    MyTuple = namedtuple('MyTuple', options)
    opt = MyTuple(**options)

    return opt

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment_dir")

    parser.add_argument("--epoch", type=str, default='latest')
    parser.add_argument("--output_name", type=str, default='')

    args = parser.parse_args()

    return (args.experiment_dir, args.epoch, args.output_name)

if __name__ == '__main__':
    arguments = parse_arguments()
    main(*arguments)


