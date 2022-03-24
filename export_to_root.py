import os

import numpy as np
import torch, yaml, ROOT

from model import *
from datasets import *

def main(opt):
  dataset_test = CustomDatasetDataLoader(opt, valid=True).load_data()
  dataset_test_size = len(dataset_test)
  print("Number of test images={}".format(dataset_test_size))

  model = Pix2pix(opt)
  print("model {} was created".format(type(model).__name__))
  model.setup(opt)
  model.eval()

  ROOT.gROOT.ProcessLine("struct Digs { Int_t ch; std::vector<short> digvec; };")
  ROOT.gROOT.ProcessLine("struct packet { Int_t ch; Int_t tick; int_t adc; };")

  f = ROOT.TFile.Open(opt.out_path, "RECREATE")
  t = ROOT.TTree("digs_hits", "ndfdtranslations")

  rawdigits_pred = ROOT.vector("Digs")(480)
  t.Branch("rawdigits_translated", rawdigits_pred)
  rawdigits_true = ROOT.vector("Digs")(480)
  t.Branch("rawdigits_true", rawdigits_true)
  packets = ROOT.vector("packet")()
  t.Branch("nd_packets", packets)

  for i, data in enumerate(dataset_test):
    rawdigits_pred.clear()
    rawdigits_true.clear()
    packet.clear()

    model.set_input(data)
    model.test()

    visuals = model.get_current_visuals()
    ch_offset, tick_offset = opt.channel_offset, opt.tick_offset
    realA = visuals['real_A'].cpu()[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset]
    realB = visuals['real_B'].cpu()[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset]
    fakeB = visuals['fake_B'].cpu()[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset]
    realA = realA[0, 0].numpy().astype(int) # Nd packets adc
    realB = realB[0, 0].numpy().astype(int) # true fd response to nd event
    fakeB = fakeB[0, 0].numpy().astype(int) # pred fd response translated from nd event
    
    print(realA.shape, realB.shape, fakeB.shape, sep='\n')

    for ch, adc_vec in enumerate(realB):
      digvec = ROOT.vector("short")(6000)
      for tick, adc in enumerate(adc_vec):
        digvec[tick] = adc
      
      rawdigits_true[ch].ch = ch + opt.first_ch_number
      rawdigits_true[ch].digvec = digvec

    for ch, adc_vec in enumerate(fakeB):
      digvec = ROOT.vector("short")(6000)
      for tick, adc in enumerate(adc_vec):
        digvec[tick] = adc
      
      rawdigits_pred[ch].ch = ch + opt.first_ch_number
      rawdigits_pred[ch].digvec = digvec 

    for ch, adc_vec in enumerate(realA):
      for tick, adc in enumerate(adc_vec):
        if adc != 0:
          packet = ROOT.packet()
          packet.ch = ch
          packet.tick = tick
          packet.adc = adc
          packets.push_back(packet)
    
    t.Fill()

  f.Write()
  f.Close()

if __name__ == '__main__':
  experiment_dir = '/home/awilkins/extrapolation_pix2pix/checkpoints/nd_fd_radi_1-8_vtxaligned_noped_morechannels_fddriftfixed_14'

  with open(os.path.join(experiment_dir, 'config.yaml')) as f:
    options = yaml.load(f, Loader=yaml.FullLoader)

  # If data is not on the current node, grab it from the share disk.
  if not os.path.exists(options['dataroot']):
    options['dataroot'] = options['dataroot_shared_disk']

  options['gpu_ids'] = [0]
  # For resnet dropout is in the middle of a sequential so needs to be commented out to maintain layer indices
  # For for unet its at the end so can remove it and still load the state_dict (nn.Dropout has no weights so
  # we don't get an unexpected key error when doing this)
  # options['no_dropout'] = True
  options['num_threads'] = 1
  options['phase'] = 'test'
  options['isTrain'] = False
  options['epoch'] = 'bias_good_mu_best_sigma' # 'latest', 'best_{bias_mu, bias_sigma, loss_pix, loss_channel}', 'bias_good_mu_best_sigma'

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

  options['out_path'] = "/state/partition1/awilkins/test.root"
  options['first_ch_number'] = 14400

  print("Using configuration:")
  for key, value in options.items():
    print("{}={}".format(key, value))

  MyTuple = namedtuple('MyTuple', options)
  opt = MyTuple(**options)

  main(opt)