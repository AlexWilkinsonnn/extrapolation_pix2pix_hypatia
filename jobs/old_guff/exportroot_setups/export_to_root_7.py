import os, sys, re
from collections import namedtuple

import numpy as np
import sparse
import torch, yaml, ROOT
from array import array
from tqdm import tqdm
from matplotlib import pyplot as plt

sys.path.append('/home/awilkins/extrapolation_pix2pix')
from model import *
from dataset import *

def main(optZ, optU, optV, opt_all):
    dataset_testZ = CustomDatasetDataLoader(optZ, valid=opt_all.valid).load_data()
    dataset_testZ_size = len(dataset_testZ)
    print("Number of test images Z = {}".format(dataset_testZ_size))

    modelZ = Pix2pix(optZ)
    print("model {} was created".format(type(modelZ).__name__))
    modelZ.setup(optZ)
    modelZ.eval()

    dataset_testU = CustomDatasetDataLoader(optU, valid=opt_all.valid).load_data()
    dataset_testU_size = len(dataset_testU)
    print("Number of test images U = {}".format(dataset_testU_size))

    modelU = Pix2pix(optU)
    print("model {} was created".format(type(modelU).__name__))
    modelU.setup(optU)
    modelU.eval()

    dataset_testV = CustomDatasetDataLoader(optV, valid=opt_all.valid).load_data()
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
        if opt_all.start_i != -1 and i < opt_all.start_i:
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
            if opt_all.convert_channelsZ and data['A'].size()[2] == 480:
                data['A'] = data['A'][:, :5, :, :]

                double_columns = [0,4,7,12,15,19,23,27,31,34,38,42,46,50,54,57,61,76,80,84,88,92,
                    95,99,103,107,111,115,118,122,126,130,134,138,141,145,149,153,157,161,164,168,
                    172,176,180,184,187,191,195,199,203,207,211,214,218,222,226,230,234,237,241,245,
                    249,253,257,260,264,268,272,287,291,295,298,302,306,310,314,318,321,325,329,333,
                    337,341,344,348,352,356,360,364,367,371,375,379,383,387,391,394,398,402,406,410,
                    414,417,421,425,429,433,437,440,444,448,452,456,460,463,467,471,475,479]
                for ch, chVec in enumerate(data['A'][0][4]):
                    if ch in double_columns:
                        chVec[:] = 1.0
                    else:
                        chVec[:] = 0.0

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

            if opt.apply_mask:
                if ch_offset and tick_offset:
                    mask = data['mask'].cpu()[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset].numpy().astype(bool)[0, 0]
                else:
                    mask = data['mask'].cpu().numpy().astype(bool)[0, 0]

                fakeB = fakeB * mask
            
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
    experiment_dirZ = '/home/awilkins/extrapolation_pix2pix/checkpoints/nd_fd_radi_1-8_vtxaligned_noped_morechannels_fddriftfixed_14'
    with open(os.path.join(experiment_dirZ, 'config.yaml')) as f:
        optionsZ = yaml.load(f, Loader=yaml.FullLoader)
    optionsZ['first_ch_number'] = 14400 # T10P2 (Z): 14400, T10P1 (V): 13600, T10P0 (U): 12800
    optionsZ['num_channels'] = 480 # 480, 800
    optionsZ['epoch'] = 'latest' # 'latest', 'best_{bias_mu, bias_sigma, loss_pix, loss_channel}', 'bias_good_mu_best_sigma'
    optionsZ['apply_mask'] = False # Zero out non signal region with the mask (temporary deal with noise artifacts I am seeing atm)

    optionsZ['dataroot'] = '/state/partition1/awilkins/nd_fd_radi_geomservice_Z_wiredistance_UVZvalid'
    optionsZ['dataroot_shared_disk'] = '/share/gpu3/awilkins/nd_fd_radi_geomservice_Z_wiredistance_UVZvalid' 
    optionsZ['nd_sparse'] = True
    optionsZ['input_nc'] = 5
    optionsZ['channel_offset'] = 0
    optionsZ['tick_offset'] = 0

    experiment_dirU = '/home/awilkins/extrapolation_pix2pix/checkpoints/nd_fd_radi_geomservice_U_wiredistance_UVZvalid_1'
    with open(os.path.join(experiment_dirU, 'config.yaml')) as f:
        optionsU = yaml.load(f, Loader=yaml.FullLoader)
    optionsU['first_ch_number'] = 12800 # T10P2 (Z): 14400, T10P1 (V): 13600, T10P0 (U): 12800
    optionsU['num_channels'] = 800 # 480, 800
    optionsU['epoch'] = 'best_bias_mu' # 'latest', 'best_{bias_mu, bias_sigma, loss_pix, loss_channel}', 'bias_good_mu_best_sigma'
    optionsU['apply_mask'] = True

    experiment_dirV = '/home/awilkins/extrapolation_pix2pix/checkpoints/nd_fd_radi_geomservice_V_wiredistance_UVZvalid_2'
    with open(os.path.join(experiment_dirV, 'config.yaml')) as f:
        optionsV = yaml.load(f, Loader=yaml.FullLoader)
    optionsV['first_ch_number'] = 13600 # T10P2 (Z): 14400, T10P1 (V): 13600, T10P0 (U): 12800
    optionsV['num_channels'] = 800 # 480, 800
    optionsV['epoch'] = 'best_bias_mu' # 'latest', 'best_{bias_mu, bias_sigma, loss_pix, loss_channel}', 'bias_good_mu_best_sigma'
    optionsV['apply_mask'] = True

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
    options_all['out_path'] = "/state/partition1/awilkins/ndfdT10UVZvalid_Zdoublecols14latest_Ugeomservice1biasmu_Vgeomservice2biasmu_fdtruefdpredndin_valid4202_3000-3500.root"

    options_all['start_i'] = 3000  # -1 for all
    options_all['end_i'] = 3500 # -1 for all

    options_all['valid'] = True

    options_all['convert_channelsZ'] = True # Convert wire distance Z data to the 5 channel data with the channel indicating if two columns are mapped to a channel

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

