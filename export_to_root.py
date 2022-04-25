import os, sys
from collections import namedtuple

import numpy as np
import torch, yaml, ROOT
from tqdm import tqdm

from model import *
from dataset import *

def main(opt):
  dataset_test = CustomDatasetDataLoader(opt, valid=valid).load_data()
  dataset_test_size = len(dataset_test)
  print("Number of test images={}".format(dataset_test_size))

  model = Pix2pix(opt)
  print("model {} was created".format(type(model).__name__))
  model.setup(opt)
  model.eval()

  # ROOT.gROOT.ProcessLine("struct Digs { Int_t ch; std::vector<short> digvec; };")
  # ROOT.gROOT.ProcessLine("struct Packet { Int_t ch; Int_t tick; Int_t adc; };")
  ROOT.gROOT.ProcessLine("#include <vector>")

  f = ROOT.TFile.Open(opt.out_path, "RECREATE")
  t = ROOT.TTree("digs_hits", "ndfdtranslations")

  chs = ROOT.vector("int")(480)
  t.Branch("channels", chs)
  digs_pred = ROOT.vector("std::vector<int>")(480)
  t.Branch("rawdigits_translated", digs_pred)
  digs_true = ROOT.vector("std::vector<int>")(480)
  t.Branch("rawdigits_true", digs_true)
  packets = ROOT.vector("std::vector<int>")()
  t.Branch("nd_packets", packets)

  for i in range(480):
    chs[i] = i + opt.first_ch_number

  for i, data in enumerate(dataset_test):
    if opt.end_i != -1 and i < opt.start_i:
      continue
    if opt.end_i != -1 and i >= opt.end_i:
      break

    for i in range(digs_true.size()):
      digs_true[i].clear()
      digs_pred[i].clear()
    packets.clear()

    model.set_input(data)
    model.test()

    visuals = model.get_current_visuals()
    ch_offset, tick_offset = opt.channel_offset, opt.tick_offset
    realA = visuals['real_A'].cpu()[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset]
    realB = visuals['real_B'].cpu()[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset]
    fakeB = visuals['fake_B'].cpu()[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset]
    realA/=opt.A_ch0_scalefactor
    realB/=opt.B_ch0_scalefactor
    fakeB/=opt.B_ch0_scalefactor
    realA = realA[0, 0].numpy().astype(int) # Nd packets adc
    realB = realB[0, 0].numpy().astype(int) # true fd response to nd event
    fakeB = fakeB[0, 0].numpy().astype(int) # pred fd response translated from nd event
    
    for ch, adc_vec in enumerate(realB):
      digs_true[ch] = ROOT.vector("int")(4492)
      for tick, adc in enumerate(adc_vec):
        digs_true[ch][tick] = int(adc)

    for ch, adc_vec in enumerate(fakeB):
      digs_pred[ch] = ROOT.vector("int")(4492)
      for tick, adc in enumerate(adc_vec):
        digs_pred[ch][tick] = int(adc)
      
    for ch, adc_vec in enumerate(realA):
      for tick, adc in enumerate(adc_vec):
        if adc != 0:
          packet = ROOT.vector("int")(3)
          packet[0] = ch + opt.first_ch_number
          packet[1] = tick
          packet[2] = adc
          packets.push_back(packet)

    t.Fill()

  f.Write()
  f.Close()

if __name__ == '__main__':
  experiment_dir = '/home/awilkins/extrapolation_pix2pix/checkpoints/nd_fd_radi_1-8_vtxaligned_noped_morechannels_fddriftfixed_14'

  with open(os.path.join(experiment_dir, 'config.yaml')) as f:
    options = yaml.load(f, Loader=yaml.FullLoader)

  # options['out_path'] = "/state/partition1/awilkins/nd_fd_radi_1-8_vtxaligned_noped_morechannels_fddriftfixed_14_latest_T10P2_fdtrue_fdpred_ndin_train9500-19000.root"
  options['out_path'] = "/state/partition1/awilkins/test.root"
  options['first_ch_number'] = 14400

  options['start_i'] = 9500 # -1 for all
  options['end_i'] = 19000 # -1 for all
  options['serial_batches'] = True # turn off shuffle so can split the work up

  valid = False
  options['phase'] = 'train' # 'train', 'test'

  # If data is not on the current node, grab it from the share disk.
  if not os.path.exists(options['dataroot']):
    options['dataroot'] = options['dataroot_shared_disk']

  options['gpu_ids'] = [0]
  # For resnet dropout is in the middle of a sequential so needs to be commented out to maintain layer indices
  # For for unet its at the end so can remove it and still load the state_dict (nn.Dropout has no weights so
  # we don't get an unexpected key error when doing this)
  # options['no_dropout'] = True
  options['num_threads'] = 1
  options['isTrain'] = False
  options['epoch'] = 'latest' # 'latest', 'best_{bias_mu, bias_sigma, loss_pix, loss_channel}', 'bias_good_mu_best_sigma'

  half_precision = False
  if half_precision:
    print("###########################################\n" + 
      "Using FP16" +
      "\n###########################################")

  # have replaced valid with a few files of interest
  test_sample = False
  if test_sample:
    print("###########################################\n" + 
        "Using test_sample" +
        "\n###########################################")

  print("Using configuration:")
  for key, value in options.items():
    print("{}={}".format(key, value))

  MyTuple = namedtuple('MyTuple', options)
  opt = MyTuple(**options)

  main(opt)
