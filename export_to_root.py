import os, sys, re
from collections import namedtuple

import numpy as np
import sparse
import torch, yaml, ROOT
from array import array
from tqdm import tqdm
from matplotlib import pyplot as plt

from model import *
from dataset import *

def main(optZ, optU, optV, opt_all):
    dataset_testZ = CustomDatasetDataLoader(optZ, valid=valid).load_data()
    dataset_testZ_size = len(dataset_testZ)
    print("Number of test images Z = {}".format(dataset_testZ_size))

    modelZ = Pix2pix(optZ)
    print("model {} was created".format(type(modelZ).__name__))
    modelZ.setup(optZ)
    modelZ.eval()

    dataset_testU = CustomDatasetDataLoader(optU, valid=valid).load_data()
    dataset_testU_size = len(dataset_testU)
    print("Number of test images U = {}".format(dataset_testU_size))

    modelU = Pix2pix(optU)
    print("model {} was created".format(type(modelU).__name__))
    modelU.setup(optU)
    modelU.eval()

    dataset_testV = CustomDatasetDataLoader(optV, valid=valid).load_data()
    dataset_testV_size = len(dataset_testV)
    print("Number of test images V = {}".format(dataset_testV_size))

    modelV = Pix2pix(optV)
    print("model {} was created".format(type(modelV).__name__))
    modelV.setup(optV)
    modelV.eval()

    # ROOT.gROOT.ProcessLine("struct Digs { Int_t ch; std::vector<short> digvec; };")
    # ROOT.gROOT.ProcessLine("struct Packet { Int_t ch; Int_t tick; Int_t adc; };")
    ROOT.gROOT.ProcessLine("#include <vector>")

    f = ROOT.TFile.Open(opt_all.out_path, "RECREATE")
    t = ROOT.TTree("digs_hits", "ndfdtranslations")

    total_num_channels = int(optZ.num_channels + optU.num_channels + optV.num_channels)

    chs = ROOT.vector("int")(total_num_channels)
    t.Branch("channels", chs)
    digs_pred = ROOT.vector("std::vector<int>")(total_num_channels)
    t.Branch("rawdigits_translated", digs_pred)
    digs_true = ROOT.vector("std::vector<int>")(total_num_channels)
    t.Branch("rawdigits_true", digs_true)
    packets = ROOT.vector("std::vector<int>")()
    t.Branch("nd_packets", packets)
    ev_num = array('i', [0])
    t.Branch("ev_num", ev_num, 'ev_num/I')

    ch_to_vecnum = {}
    cntr = 0
    for opt in [optZ, optU, optV]:
        for i in range(opt.num_channels):
            chs[cntr] = i + opt.first_ch_number
            ch_to_vecnum[i + opt.first_ch_number] = cntr
            cntr += 1

    for i, (dataZ, dataU, dataV) in enumerate(zip(dataset_testZ, dataset_testU, dataset_testV)):
        if opt_all.end_i != -1 and i < opt_all.start_i:
            continue
        if opt_all.end_i != -1 and i >= opt_all.end_i:
            break

        for i in range(digs_true.size()):
            digs_true[i].clear()
            digs_pred[i].clear()
        packets.clear()
        ev_num[0] = -1

        data_name = os.path.basename(dataZ['A_paths'][0])
        ev_num[0] = int(re.match("([0-9]+)", data_name)[0])

        ev_numU = int(re.match("([0-9]+)", os.path.basename(dataU['A_paths'][0]))[0])
        ev_numV = int(re.match("([0-9]+)", os.path.basename(dataV['A_paths'][0]))[0])
        if ev_num[0] != ev_numU or ev_num[0] != ev_numV:
            print(os.path.basename(dataV['A_paths'][0]), os.path.basename(dataV['A_paths'][0]))
            raise Exception("brokey")

        for data, model, opt in zip([dataZ, dataU, dataV], [modelZ, modelU, modelV], [optZ, optU, optV]):
            model.set_input(data)
            model.test()

            visuals = model.get_current_visuals()
            ch_offset, tick_offset = opt.channel_offset, opt.tick_offset
            if ch_offset and tick_offset:
                realA = visuals['real_A'].cpu()[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset]
                realB = visuals['real_B'].cpu()[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset]
                fakeB = visuals['fake_B'].cpu()[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset]
            else:
                realA = visuals['real_A'].cpu()
                realB = visuals['real_B'].cpu()
                fakeB = visuals['fake_B'].cpu()
            realA/=opt.A_ch0_scalefactor
            realB/=opt.B_ch0_scalefactor
            fakeB/=opt.B_ch0_scalefactor
            realA = realA[0, 0].numpy().astype(int) # Nd packets adc
            realB = realB[0, 0].numpy().astype(int) # true fd response to nd event
            fakeB = fakeB[0, 0].numpy().astype(int) # pred fd response translated from nd event
            
            for ch_local, adc_vec in enumerate(realB):
                ch = ch_to_vecnum[ch_local + opt.first_ch_number]
                digs_true[ch] = ROOT.vector("int")(4492)
                for tick, adc in enumerate(adc_vec):
                    digs_true[ch][tick] = int(adc)

            for ch_local, adc_vec in enumerate(fakeB):
                ch = ch_to_vecnum[ch_local + opt.first_ch_number]
                digs_pred[ch] = ROOT.vector("int")(4492)
                for tick, adc in enumerate(adc_vec):
                    digs_pred[ch][tick] = int(adc)
                
            for ch_local, adc_vec in enumerate(realA):
                for tick, adc in enumerate(adc_vec):
                    if adc != 0:
                        packet = ROOT.vector("int")(3)
                        packet[0] = ch_local + opt.first_ch_number
                        packet[1] = tick
                        packet[2] = adc
                        packets.push_back(packet)

        t.Fill()

    f.Write()
    f.Close()

if __name__ == '__main__':
    experiment_dirZ = '/home/awilkins/extrapolation_pix2pix/checkpoints/nd_fd_radi_geomservice_Z_wiredistance_UVZvalid_1'
    with open(os.path.join(experiment_dirZ, 'config.yaml')) as f:
        optionsZ = yaml.load(f, Loader=yaml.FullLoader)
    optionsZ['first_ch_number'] = 14400 # T10P2 (Z): 14400, T10P1 (V): 13600, T10P0 (U): 12800
    optionsZ['num_channels'] = 480 # 480, 800
    optionsZ['epoch'] = 'best_loss_pix' # 'latest', 'best_{bias_mu, bias_sigma, loss_pix, loss_channel}', 'bias_good_mu_best_sigma'

    experiment_dirU = '/home/awilkins/extrapolation_pix2pix/checkpoints/nd_fd_radi_geomservice_U_wiredistance_UVZvalid_1'
    with open(os.path.join(experiment_dirU, 'config.yaml')) as f:
        optionsU = yaml.load(f, Loader=yaml.FullLoader)
    optionsU['first_ch_number'] = 12800 # T10P2 (Z): 14400, T10P1 (V): 13600, T10P0 (U): 12800
    optionsU['num_channels'] = 800 # 480, 800
    optionsU['epoch'] = 'best_loss_pix' # 'latest', 'best_{bias_mu, bias_sigma, loss_pix, loss_channel}', 'bias_good_mu_best_sigma'

    experiment_dirV = '/home/awilkins/extrapolation_pix2pix/checkpoints/nd_fd_radi_geomservice_V_wiredistance_UVZvalid_1'
    with open(os.path.join(experiment_dirV, 'config.yaml')) as f:
        optionsV = yaml.load(f, Loader=yaml.FullLoader)
    optionsV['first_ch_number'] = 13600 # T10P2 (Z): 14400, T10P1 (V): 13600, T10P0 (U): 12800
    optionsV['num_channels'] = 800 # 480, 800
    optionsV['epoch'] = 'best_loss_pix' # 'latest', 'best_{bias_mu, bias_sigma, loss_pix, loss_channel}', 'bias_good_mu_best_sigma'

    for options in [optionsZ, optionsU, optionsV]:
        options['gpu_ids'] = [0]
        options['phase'] = 'train' # 'train', 'test'
        options['num_threads'] = 1
        options['serial_batches'] = True
        options['isTrain'] = False
        # If data is not on the current node, grab it from the share disk.
        if not os.path.exists(options['dataroot']):
            options['dataroot'] = options['dataroot_shared_disk']
        # For resnet dropout is in the middle of a sequential so needs to be commented out to maintain layer indices
        # For for unet its at the end so can remove it and still load the state_dict (nn.Dropout has no weights so
        # we don't get an unexpected key error when doing this)
        # options['no_dropout'] = True

    options_all = {}

    # options_all['out_path'] = "/state/partition1/awilkins/nd_fd_radi_geomservice_Z_wiredistance_8_losspix_T10P2_fdtrue_fdpred_ndin_valid4202.root"
    options_all['out_path'] = "/state/partition1/awilkins/UVZtest10.root"

    options_all['start_i'] = 0  # -1 for all
    options_all['end_i'] = 10 # -1 for all

    valid = True

    print("Using Z configuration:")
    for key, value in optionsZ.items():
        print("{}={}".format(key, value))
    print("Using U configuration:")
    for key, value in optionsU.items():
        print("{}={}".format(key, value))
    print("Using V configuration:")
    for key, value in optionsV.items():
        print("{}={}".format(key, value))

    MyTupleZ = namedtuple('MyTupleZ', optionsZ)
    optZ = MyTupleZ(**optionsZ)
    MyTupleU = namedtuple('MyTupleU', optionsU)
    optU = MyTupleU(**optionsU)
    MyTupleV = namedtuple('MyTupleV', optionsV)
    optV = MyTupleV(**optionsV)
    MyTuple_all = namedtuple('MyTuple_all', options_all)
    opt_all = MyTuple_all(**options_all)

    main(optZ, optU, optV, opt_all)

