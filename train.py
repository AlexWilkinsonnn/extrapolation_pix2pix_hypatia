import argparse, time, os, sys
from collections import namedtuple

import numpy as np
from scipy import stats
import torch
import yaml

from model import *
from dataset import *
from networks import CustomLoss


def main(opt):
    dataset = CustomDatasetDataLoader(opt).load_data()
    dataset_size = len(dataset)
    print("Number of training images = %d" % dataset_size)

    dataset_valid = CustomDatasetDataLoader(opt, valid=True).load_data()
    dataset_valid_size = len(dataset_valid)
    print("Number of validation images = %d" % dataset_valid_size)
    dataset_valid_iterator = iter(dataset_valid)

    model = Pix2pix(opt)
    print("model [%s] was created" % type(model).__name__)
    model.setup(opt)
    total_iters = 0

    best_metrics = {}

    for epoch in range(1, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        model.update_learning_rate()
        
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            for tile_number in range(1):#len(data['A'])):
                # mask = data['mask'][tile_number]
                # if opt.using_mask:
                #     mask.requires_grad = False
                # tile_data = { 'A' : data['A'][tile_number], 'B' : data['B'][tile_number], 
                #     'A_paths' : data['A_paths'], 'B_paths': data['B_paths'], 
                #     'mask' : mask }
                tile_data = data
                tile_data['mask'].requires_grad = False
                # print("{} {} {} {}".format(tile_data['A'].size(), tile_data['B'].size(), tile_data['mask'].size(), tile_data['A_paths'])) 
                        
                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
                model.set_input(tile_data)
                model.optimize_parameters()

                if total_iters % opt.display_freq == 0:
                    visuals = model.get_current_visuals()
                    
                    image_realA = visuals['real_A'][0].data # Taking first in the batch to save
                    image_realA[0]/=opt.A_ch0_scalefactor
                    # image_realA[1]/=opt.A_ch1_scalefactor
                    arr_realA = image_realA.cpu().float().numpy().astype(int)
                    np.save(os.path.join(opt.checkpoints_dir, opt.name, "realA.npy"), arr_realA)
                
                    image_realB = visuals['real_B'][0].data
                    image_realB[0]/=opt.B_ch0_scalefactor
                    arr_realB = image_realB.cpu().float().numpy().astype(int)
                    np.save(os.path.join(opt.checkpoints_dir, opt.name, "realB.npy"), arr_realB)
                
                    image_fakeB = visuals['fake_B'][0].data
                    image_fakeB[0]/=opt.B_ch0_scalefactor
                    arr_fakeB = image_fakeB.cpu().float().numpy().astype(int)
                    np.save(os.path.join(opt.checkpoints_dir, opt.name, "fakeB.npy"), arr_fakeB)

                    losses = model.get_current_losses()
                    loss_line = "total_iters={}, epoch={}, epoch_iter={} : G_GAN={}, G_pix={}, G_channel={}, D_real={}, D_fake={}".format(
                        total_iters, epoch, epoch_iter, losses['G_GAN'], losses['G_pix'],
                        losses['G_channel'], losses['D_real'], losses['D_fake'])
                    with open(os.path.join(opt.checkpoints_dir, opt.name, "loss.txt"), 'a') as f:
                        f.write(loss_line + '\n')
                    
                if total_iters % opt.print_freq == 0: # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    loss_line = "total_iters={}, epoch={}, epoch_iter={} : G_GAN={}, G_pix={}, G_channel={}, D_real={}, D_fake={}".format(
                        total_iters, epoch, epoch_iter, losses['G_GAN'], losses['G_pix'],
                        losses['G_channel'], losses['D_real'], losses['D_fake'])
                    print(loss_line)
                    with open(os.path.join(opt.checkpoints_dir, opt.name, "loss.txt"), 'a') as f:
                        f.write(loss_line + '\n')

                if total_iters % opt.save_latest_freq == 0: # cache our latest model every <save_latest_freq> iterations
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'latest'
                    model.save_networks(save_suffix)

                iter_data_time = time.time()

                if opt.valid_freq != 'epoch' and total_iters % opt.valid_freq == 0:
                    valid(dataset_valid_iterator, dataset_valid, model, opt, epoch, total_iters, best_metrics)

        if epoch % opt.save_epoch_freq == 0: # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

        if opt.valid_freq == 'epoch':
            valid(dataset_valid_iterator, dataset_valid, model, opt, epoch, total_iters)


def valid(dataset_itr, dataset, model, opt, epoch, total_itrs, best_metrics):
    model.eval()

    G_pix_losses, G_channel_losses = [], []
    losses_event_over20, losses_event_over20_fractional = [], []
    losses_event_underneg20, losses_event_underneg20_fractional = [], []
    for i in range(len(dataset)): # Run the generator on some validation images for visualisation
        try:
            data = next(dataset_itr)

        except StopIteration:
            dataset_itr = iter(dataset)
            data = next(dataset_itr)

        # data = { 'A' : data['A'][0], 'B' : data['B'][0], 'A_paths': data['A_paths'], 'B_paths': data['B_paths'],
        #     'mask' : data['mask'][0] }
        model.set_input(data)
        model.test()

        visuals = model.get_current_visuals()
        ch_offset, tick_offset = opt.channel_offset, opt.tick_offset
        realA = visuals['real_A'].cpu()[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset] 
        realB = visuals['real_B'].cpu()[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset]
        fakeB = visuals['fake_B'].cpu()[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset]
        realB/=opt.B_ch0_scalefactor
        fakeB/=opt.B_ch0_scalefactor
        mask = data['mask'].cpu()[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset]
        loss_pix, loss_channel = CustomLoss(realA, fakeB, realB, 'AtoB', mask, opt.B_ch0_scalefactor, opt.mask_type, opt.nonzero_L1weight, opt.rms)
        loss_event_over20 = (fakeB.float() * (fakeB.float() > 20)).sum() - (realB.float() * (realB.float() > 20)).sum()
        loss_event_over20_fractional = loss_event_over20/((realB.float() * (realB.float() > 20)).sum())
        if opt.channel_offset == 112: # induction view 
            loss_event_underneg20 = (fakeB.float() * (fakeB.float() < -20)).sum() - (realB.float() * (realB.float() < -20)).sum()
            loss_event_underneg20_fractional = loss_event_underneg20/((realB.float() * (realB.float() < -20)).sum())           
        if loss_pix != 0:
            G_pix_losses.append(loss_pix.item())
            G_channel_losses.append(loss_channel.item())
            if (realB.float() * (realB.float() > 20)).sum() != 0:
                losses_event_over20.append(loss_event_over20.item())
                losses_event_over20_fractional.append(loss_event_over20_fractional.item())
            if opt.channel_offset == 112 and (realB.float() * (realB.float() < -20)).sum() != 0:
                losses_event_underneg20.append(loss_event_underneg20.item())
                losses_event_underneg20_fractional.append(loss_event_underneg20_fractional.item())

        if i < 10:
            realA = realA[0] # Remove batch dimension (batchsize = 1 for valid)
            realA[0]/=opt.A_ch0_scalefactor
            # realA[1]/=opt.A_ch1_scalefactor
            arr_realA = realA.numpy().astype(int)
            np.save(os.path.join(opt.checkpoints_dir, opt.name, "realA_valid{}.npy".format(i)), arr_realA)
        
            arr_realB = realB[0].numpy().astype(int)
            np.save(os.path.join(opt.checkpoints_dir, opt.name, "realB_valid{}.npy".format(i)), arr_realB)
        
            arr_fakeB =fakeB[0].numpy().astype(int)
            np.save(os.path.join(opt.checkpoints_dir, opt.name, "fakeB_valid{}.npy".format(i)), arr_fakeB)

    if opt.channel_offset == 112: # induction view
        bias_mu_over20, bias_sigma_over20 = stats.norm.fit(losses_event_over20_fractional)
        bias_mu_underneg20, bias_sigma_underneg20 = stats.norm.fit(losses_event_underneg20_fractional)
        bias_mu = (abs(float(bias_mu_over20)) + abs(float(bias_mu_underneg20)))/2
        bias_sigma = (abs(float(bias_sigma_over20)) + abs(float(bias_sigma_underneg20)))/2
    else:
        bias_mu, bias_sigma = stats.norm.fit(losses_event_over20_fractional)
        bias_mu, bias_sigma = float(bias_mu), float(bias_sigma)
    loss_pix, loss_channel = float(np.mean(G_pix_losses)), float(np.mean(G_channel_losses))
    
    if 'bias_mu' not in best_metrics:
        best_metrics['bias_mu'] = bias_mu
        best_metrics['bias_mu_itr'] = total_itrs
        model.save_networks("best_bias_mu")
    elif abs(best_metrics['bias_mu']) > abs(bias_mu):
        best_metrics['bias_mu'] = bias_mu
        best_metrics['bias_mu_itr'] = total_itrs
        model.save_networks("best_bias_mu")

    if 'bias_sigma' not in best_metrics:
        best_metrics['bias_sigma'] = bias_sigma
        best_metrics['bias_sigma_itr'] = total_itrs
        model.save_networks("best_bias_sigma")
    elif best_metrics['bias_sigma'] > bias_sigma:
        best_metrics['bias_sigma'] = bias_sigma
        best_metrics['bias_sigma_itr'] = total_itrs
        model.save_networks("best_bias_sigma")

    if abs(bias_mu) < abs(0.05):
        if 'bias_good_mu_best_sigma' not in best_metrics:
            best_metrics['bias_good_mu_best_sigma'] = (bias_mu, bias_sigma)
            best_metrics['bias_good_mu_best_sigma_itr'] = total_itrs
            model.save_networks("bias_good_mu_best_sigma")
        elif best_metrics['bias_good_mu_best_sigma'][1] > bias_sigma:
            best_metrics['bias_good_mu_best_sigma'] = (bias_mu, bias_sigma)
            best_metrics['bias_good_mu_best_sigma_itr'] = total_itrs
            model.save_networks("bias_good_mu_best_sigma")

    if 'loss_pix' not in best_metrics:
        best_metrics['loss_pix'] = loss_pix
        best_metrics['loss_pix_itr'] = total_itrs
        model.save_networks("best_loss_pix")
    elif best_metrics['loss_pix'] > loss_pix:
        best_metrics['loss_pix'] = loss_pix
        best_metrics['loss_pix_itr'] = total_itrs
        model.save_networks("best_loss_pix")        

    if 'loss_channel' not in best_metrics:
        best_metrics['loss_channel'] = loss_channel
        best_metrics['loss_channel_itr'] = total_itrs
        model.save_networks("best_loss_channel")
    elif best_metrics['loss_channel'] > loss_channel:
        best_metrics['loss_channel'] = loss_channel
        best_metrics['loss_channel_itr'] = total_itrs
        model.save_networks("best_loss_channel")        
        
    print(best_metrics)
        
    with open(os.path.join(options['checkpoints_dir'], options['name'], "best_metrics.yaml"), 'w') as f:
        yaml.dump(best_metrics, f)

    loss_line = "VALID:total_iters={}, epoch={} : G_pix={}, G_channel={}".format(
        total_itrs, epoch, np.mean(G_pix_losses), np.mean(G_channel_losses))
    print(loss_line)
    with open(os.path.join(opt.checkpoints_dir, opt.name, "loss.txt"), 'a') as f:
        f.write(loss_line + '\n')     
    
    model.train()


if __name__ == '__main__':
    options = {
        'dataroot' : '/state/partition1/awilkins/depos_X_4492_U_fsb_fixedbb_nu',
        'full_image' : False, # True if you want to crop a full image into 512 tiles, false otherwise
        'samples' : 1, # 0 to do samples = ticks//512
        'mask_type' : 'auto', # 'auto', 'saved', 'none'. 'none_weighted', 'saved_1rms'
        'rms' : 3.610753167639414, # needed is mask_type='saved_1rms'. collection_fsb_nu: 3.610753167639414, U_fsb_fixedbb_nu: 3.8106195813271166, V_fsb_fixedbb_nu: 3.8106180475002605
        # 'A_ch0_scalefactor' : 5.809056318801011e-05, # collection:0.00012918744969763678, U:5.809056318801011e-05, V:0.0001036767938158866. charge [0,1] using 0 and maximum in train
        # 'A_ch1_scalefactor' : 0.002782244826415745, # collection:0.0027748949702253766, U:0.002782244826415745, V:0.0027785650378718413. X [0,1] using 0 and maximum in train
        # 'B_ch0_scalefactor' : 0.0004268032437046522, # collection:0.00031279324366593683, U:0.0004268032437046522, V:0.0004268032437046522. +ve adc [0,1] and -ve adc [-1,0] with same scalefactor ie. 1/max(max(abs(-ve adc)),max(+ve adc))
        # 'A_ch0_scalefactor' : 0.00012918744969763678,
        # 'A_ch1_scalefactor' : 0.0027748949702253766,
        # 'B_ch0_scalefactor' : 0.00031279324366593683,
        # 'A_ch0_scalefactor' : 0.0001036767938158866,
        # 'A_ch1_scalefactor' : 0.0027785650378718413,
        # 'B_ch0_scalefactor' : 0.0004268032437046522,
        # 'A_ch0_scalefactor' : 0.0002781641168289291, # charge is scaled to ADC in MakeSimImages so use same scaling as ADC (values > 1 possible)
        # 'A_ch1_scalefactor' : 0.0016666138905601323, # 1/600.019 for max X
        # 'B_ch0_scalefactor' : 0.00031298904538341156, # 1/3195 for collection ([-900, 3195]), used to be incorrect (0.0002781641168289291, [-500, 3595])
        # 'A_ch0_scalefactor' : 0.000425531914893617, # charge is scaled to ADC in MakeSimImages so use same scaling as ADC (values > 1 possible) 
        # 'A_ch1_scalefactor' : 0.0016666138905601323, # 1/600.019 for max X
        # 'B_ch0_scalefactor' : 0.000425531914893617, # 1/2350 for induction ([-2350, 1745])
        'A_ch0_scalefactor' : 0.0002781641168289291, # Scale down the ND adc by max of the dataset for now
        'B_ch0_scalefactor' : 0.00031298904538341156, # 1/3195 for collection ([-900, 3195]), used to be incorrect (0.0002781641168289291, [-500, 3595])
        'name' : "depos_X_4492_U_fsb_fixedbb_nu_tanhclamp_7",
        'gpu_ids' : [0],
        'checkpoints_dir' : '/home/awilkins/extrapolation_pix2pix/checkpoints',
        'input_nc' :  1,
        'output_nc' : 1,
        'ngf' : 64,
        'ndf' : 64,
        'netD' : 'n_layers', # 'basic', 'n_layers', 'pixel'
        'netG' : 'unet_256', # 'unet_256', 'unet_128', 'resnet_6blocks', 'resnet_9blocks'
        'n_layers_D' : 5, # -------------- CHANGED FROM THE USUAL 5 --------------
        'norm' : 'batch', # 'batch', 'instance', 'none'
        'init_type' : 'xavier', # 'normal', 'xavier', 'kaiming', 'orthogonal'
        'init_gain' : 0.02,
        'no_dropout' : False,
        'serial_batches' : False,
        'num_threads' : 4,
        'batch_size' : 1,
        'max_dataset_size' : 8000, # Something like this
        'display_freq' : 500,
        'print_freq' : 100,
        'valid_freq' : 4000, # 'epoch' for at the end of each epoch
        'save_latest_freq' : 4000,
        'save_epoch_freq' : 2,
        'phase' : 'train',
        'n_epochs' : 10,
        'n_epochs_decay' : 7,
        'beta1' : 0.5,
        # 'lamda_L1_reg' : 0.005, # 0 for no L1 regularisation
        'adam_weight_decay' : 0, # 0 is default, 0.001
        'lr' : 0.0002, # 0.0002, 0.00005
        'gan_mode' : 'vanilla', # 'vanilla', 'lsgan', 'wgangp
        'pool_size' : 0,
        'lr_policy' : 'linear', # 'linear', 'step', 'plateau', 'cosine'
        'lr_decay_iters' : 50,
        'isTrain' : True,
        'lambda_pix' : 1000, # 1000
        'nonzero_L1weight': 10, # used for none_weighted mask type
        'lambda_channel' : 200, # 20
        'G_output_layer' : 'tanh+clampcollection', # 'identity', 'tanh', 'linear', 'relu', 'tanh+clampcollection', 'tanh+clampinduction'
        'direction' : 'AtoB',
        'collection_crop' : False,
        'channel_offset' : 16, # Induction 112, collection 16
        'tick_offset' : 58,
        'unconditional_D' : False, # need True if we want to load into a cycleGAN setup
        'noise_layer' : False,
        'kernel_size' : 3, # 4 originally, (3,5)
        'outer_stride' : (1,3), # 2 originally
        'inner_stride_1' : (1,3) # 2 originally
    }
    # epoch : 'latest' for test

    print("Using configuration:")
    for key, value in options.items():
        print("{}={}".format(key, value))

    if not os.path.exists(os.path.join(options['checkpoints_dir'], options['name'])):
        os.makedirs(os.path.join(options['checkpoints_dir'], options['name']))
    
#    else:
#        answer = ''
#        while (answer != 'y') and (answer != 'n'):
#            answer = input("Experiment name {} already exists, maybe you forgot to change it. Continue?[y/n]"
#                .format(options["name"]))
#        if answer == 'n': sys.exit()

    with open(os.path.join(options['checkpoints_dir'], options['name'], "config.yaml"), 'w') as f:
        yaml.dump(options, f)

    MyTuple = namedtuple('MyTuple', options)  
    opt = MyTuple(**options)

    main(opt)
