import os, argparse, sys
from collections import namedtuple

import torch

import yaml
import numpy as np

from model import *
from dataset import *

# Need to use this clamp output layer to allow trace
class ClampModuleInduction(torch.nn.Module):
    def __init__(self):
        super(ClampModuleInduction, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.clamp(min=-1, max=0.7425531914893617)


class ClampModuleCollection(torch.nn.Module):
    def __init__(self):
        super(ClampModuleCollection, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.clamp(min=-0.28169014084507044, max=1)


def main(EXPERIMENT_DIR, EPOCH, OUTPUT_NAME, INDUCTION):
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

    model.set_input(data)

    input_example = { 'forward' : model.get_real_A() }

    if opt.G_output_layer.startswith('tanh+clamp'):
        if 'resnet' in opt.netG:
            G.model[-1] = ClampModuleInduction() if INDUCTION else ClampModuleCollection()
        else:
            print(G)
            raise NotImplementedError("Check where in the model the custom clamp needs to be replaced")

    print("Tracing model:\n{}".format(G))
    print("Using example input with size {}".format(input_example['forward'].size()))

    with torch.no_grad():
        traced_model = torch.jit.trace_module(G, input_example)

    traced_model.save(OUTPUT_NAME if OUTPUT_NAME else EXPERIMENT_DIR + '_{}_tracedmodel.pt'.format(opt.epoch))
    print("TorchScript summary:\n{}".format(traced_model))

def load_config(experiment_dir: str, epoch: str) -> namedtuple:
    with open(os.path.join(experiment_dir, 'config.yaml')) as f:
        options = yaml.load(f, Loader=yaml.FullLoader)

    # If data is not on the current node, grab it from the share disk.
    if not os.path.exists(options['dataroot']):
        options['dataroot'] = options['dataroot_shared_disk']

    # Settings for JIT
    options['gpu_ids'] = [0] # Tells init function not to wrap model with DataParallel
    options['num_threads'] = 1
    options['phase'] = 'test'
    options['isTrain'] = False
    options['no_DataParallel'] = True # Just by having this attribute the model knows to not wrap with DataParallel

    # Set epoch to load model at
    options['epoch'] = epoch

    # Setting newer options for legacy reasons
    if 'padding_type' not in options:
        options['padding_type'] = 'reflect'
    if 'nd_sparse' not in options:
        options['nd_sparse'] = False

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
    parser.add_argument("--induction", action='store_true')

    args = parser.parse_args()

    # Clean and prepare user input
    args.experiment_dir = os.path.normpath(args.experiment_dir)
    args.output_name = args.output_name if args.output_name else os.path.basename(args.experiment_dir) + '_' + args.epoch + '_tracedmodel.pt'

    return (args.experiment_dir, args.epoch, args.output_name, args.induction)

if __name__ == '__main__':
    arguments = parse_arguments()
    main(*arguments)


