import argparse, time, os, sys
from collections import namedtuple

import numpy as np
from scipy import stats
import torch
import yaml

sys.path.append('/home/awilkins/extrapolation_pix2pix')
from model import *
from dataset import *
from networks import CustomLoss

# torch.autograd.set_detect_anomaly(True)

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
        epoch_iter = 0
        model.update_learning_rate()
        
        for i, data in enumerate(dataset):
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
                    arr_realA = image_realA.cpu().float().numpy()
                    arr_realA[0] = arr_realA[0].astype(int)
                    np.save(os.path.join(opt.checkpoints_dir, opt.name, "realA.npy"), arr_realA[0])
                
                    image_realB = visuals['real_B'][0].data
                    image_realB[0]/=opt.B_ch0_scalefactor
                    arr_realB = image_realB.cpu().float().numpy().astype(int)
                    np.save(os.path.join(opt.checkpoints_dir, opt.name, "realB.npy"), arr_realB[0])
                
                    image_fakeB = visuals['fake_B'][0].data
                    image_fakeB[0]/=opt.B_ch0_scalefactor
                    arr_fakeB = image_fakeB.cpu().float().numpy().astype(int)
                    np.save(os.path.join(opt.checkpoints_dir, opt.name, "fakeB.npy"), arr_fakeB[0])

                    losses = model.get_current_losses()
                    loss_line = "total_iters={}, epoch={}, epoch_iter={} : G_GAN={}, G_pix={}, G_channel={}, D_real={}, D_fake={}".format(
                        total_iters, epoch, epoch_iter, losses['G_GAN'], losses['G_pix'],
                        losses['G_channel'], losses['D_real'], losses['D_fake'])
                    with open(os.path.join(opt.checkpoints_dir, opt.name, "loss.txt"), 'a') as f:
                        f.write(loss_line + '\n')
                    
                elif total_iters % opt.print_freq == 0: # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
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
    dataset_itr = iter(dataset)

    G_pix_losses, G_channel_losses = [], []
    losses_event_over20, losses_event_over20_fractional = [], []
    losses_event_underneg20, losses_event_underneg20_fractional = [], []
    for i in range(len(dataset)): # Run the generator on some validation images for visualisation
        if opt.num_valid and i > opt.num_valid:
            break

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
        if ch_offset and tick_offset:
            realA = visuals['real_A'].cpu()[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset] 
            realB = visuals['real_B'].cpu()[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset]
            fakeB = visuals['fake_B'].cpu()[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset]
            mask = data['mask'].cpu()[:, :, ch_offset:-ch_offset, tick_offset:-tick_offset]
        else:
            realA = visuals['real_A'].cpu()
            realB = visuals['real_B'].cpu()
            fakeB = visuals['fake_B'].cpu()
            mask = data['mask'].cpu()
        realB/=opt.B_ch0_scalefactor
        fakeB/=opt.B_ch0_scalefactor
        loss_pix, loss_channel = CustomLoss(realA, fakeB, realB, 'AtoB', mask, opt.B_ch0_scalefactor, opt.mask_type, opt.nonzero_L1weight, opt.rms)
        loss_event_over20 = (fakeB.float() * (fakeB.float() > 20)).sum() - (realB.float() * (realB.float() > 20)).sum()
        loss_event_over20_fractional = loss_event_over20/((realB.float() * (realB.float() > 20)).sum())
        if realA.shape[2] == 800: # induction view 
            loss_event_underneg20 = (fakeB.float() * (fakeB.float() < -20)).sum() - (realB.float() * (realB.float() < -20)).sum()
            loss_event_underneg20_fractional = loss_event_underneg20/((realB.float() * (realB.float() < -20)).sum())           
        if loss_pix != 0:
            G_pix_losses.append(loss_pix.item())
            G_channel_losses.append(loss_channel.item())
            if (realB.float() * (realB.float() > 20)).sum() != 0:
                losses_event_over20.append(loss_event_over20.item())
                losses_event_over20_fractional.append(loss_event_over20_fractional.item())
            if realA.shape[2] == 800 and (realB.float() * (realB.float() < -20)).sum() != 0:
                losses_event_underneg20.append(loss_event_underneg20.item())
                losses_event_underneg20_fractional.append(loss_event_underneg20_fractional.item())

        if i < 10:
            realA = realA[0] # Remove batch dimension (batchsize = 1 for valid)
            realA[0]/=opt.A_ch0_scalefactor
            # realA[1]/=opt.A_ch1_scalefactor
            arr_realA = realA.numpy()
            arr_realA[0] = arr_realA[0].astype(int)
            np.save(os.path.join(opt.checkpoints_dir, opt.name, "realA_valid{}.npy".format(i)), arr_realA[0])
        
            arr_realB = realB[0].numpy().astype(int)
            np.save(os.path.join(opt.checkpoints_dir, opt.name, "realB_valid{}.npy".format(i)), arr_realB[0])
        
            arr_fakeB =fakeB[0].numpy().astype(int)
            np.save(os.path.join(opt.checkpoints_dir, opt.name, "fakeB_valid{}.npy".format(i)), arr_fakeB[0])

    if realA.shape[2] == 800: # induction view
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
        'dataroot' : '/state/partition1/awilkins/nd_fd_radi_geomservice_U',
        'dataroot_shared_disk' : '/share/gpu3/awilkins/nd_fd_radi_geomservice_U', # Can be /share/gpu{0,1,2,3}
        'unaligned' : True,
        'nd_sparse' : True, # nd data is saved in sparse format using the sparse library
        'full_image' : False, # True if you want to crop a full image into 512 tiles, false otherwise
        'samples' : 1, # 0 to do samples = ticks//512
        'mask_type' : 'saved', # 'auto', 'saved', 'none'. 'none_weighted', 'saved_1rms'
        'rms' : 3.610753167639414, # needed is mask_type='saved_1rms'. collection_fsb_nu: 3.610753167639414, U_fsb_fixedbb_nu: 3.8106195813271166, V_fsb_fixedbb_nu: 3.8106180475002605
        # 'A_ch0_scalefactor' : 0.00031298904538341156, # Scale down the ND adc by max of the dataset for now
        # 'B_ch0_scalefactor' : 0.00031298904538341156, # 1/3195 for collection ([-900, 3195]), used to be incorrect (0.0002781641168289291, [-500, 3595])
        # These factors are always the same
        'A_ch1_scalefactor' : 0.14085904245475275, # nd drift distance. 1/sqrt(50.4) for max nd drift in a module.
        'A_ch2_scalefactor' : 0.05645979274839422, # fd drift distance. 1/sqrt(313.705) for max fd drift accounting for putting the vtx and 2000 and associated cuts.
        # Comment based on data set in use
        # nd_fd_radi_1-8_vtxaligned_noped_morechannels
        # 'A_ch0_scalefactor' : 0.0011947431302270011, # nd adc. 1/837 for nd ADC range in nd_fd_radi_1-8_vtxaligned_noped_morechannels [4, 837].
        # 'A_ch3_scalefactor' : 0.025, # num nd packets stacked. 1/40 for nd num packets in nd_fd_radi_1-8_vtxaligned_noped_morechannels [1, 40]
        # 'A_ch4_scalefactor' : 1.0, # if two pixel cols map to the wire
        # 'B_ch0_scalefactor' : 0.00031298904538341156, # fd adc. 1/3195 for collection ([-900, 3195]).
        # nd_fd_radi_geomservice_Z && nd_fd_geomservice_Z_wiredistance
        # 'A_ch0_scalefactor' : 0.0011695906432746538, # nd adc. 1/855 for nd ADC range in nd_fd_radi_geomservice_Z [4, 855].
        # 'A_ch3_scalefactor' : 0.023809523809523808, # num nd packets stacked. 1/42 for nd num packets in nd_fd_radi_geomservice_Z [1, 42]
        # 'A_ch4_scalefactor' : 0.023809523809523808, # num first pixel triggers. 1/42 for nd num first pixel triggers in nd_fd_radi_geomservice_Z [1, 42] 
        # 'A_ch5_scalefactor' : 4.175365344467641, # wire distance, 1/0.2395 for collection wire pitch of 0.479.
        # 'B_ch0_scalefactor' : 0.00031298904538341156, # fd adc. 1/3195 for collection ([-900, 3195]).
        # nd_fd_radi_geomservice_U && nd_fd_geomservice_U_wiredistance
        'A_ch0_scalefactor' : 0.0012484394506866417, # nd adc. 1/801 for nd ADC range in nd_fd_radi_geomservice_U [4, 801].
        'A_ch3_scalefactor' : 0.022222222222222223, # num nd packets stacked. 1/45 for nd num packets in nd_fd_radi_geomservice_U [1, 45]
        'A_ch4_scalefactor' : 0.022222222222222223, # num first pixel triggers. 1/45 for nd num first pixel triggers in nd_fd_radi_geomservice_U [1, 45]
        'A_ch5_scalefactor' : 4.285408185129634, # wire distance, 1/0.23335 for induction wire pitch of 0.4667.
        'B_ch0_scalefactor' : 0.000425531914893617, # fd adc. 1/2350 for induction ([-2350, 1745])
        # nd_fd_geomservice_V && nd_fd_geomservice_V_wiredistance
        # 'A_ch0_scalefactor' : 0.0012484394506866417, # nd adc. 1/562 for nd ADC range in nd_fd_radi_geomservice_V [4, 562].
        # 'A_ch3_scalefactor' : 0.03225806451612903, # num nd packets stacked. 1/31 for nd num packets in nd_fd_radi_geomservice_V [1, 31]
        # 'A_ch4_scalefactor' : 0.03225806451612903, # num first pixel triggers. 1/31 for nd num first pixel triggers in nd_fd_radi_geomservice_U [1, 31]
        # 'A_ch5_scalefactor' : 4.285408185129634, # wire distance, 1/0.23335 for induction wire pitch of 0.4667.
        # 'B_ch0_scalefactor' : 0.000425531914893617, # fd adc. 1/2350 for induction ([-2350, 1745])
        'name' : "nd_fd_radi_geomservice_U_test",
        'gpu_ids' : [0],
        'checkpoints_dir' : '/home/awilkins/extrapolation_pix2pix/checkpoints',
        'input_nc' :  6,
        'output_nc' : 1,
        'ngf' : 64,
        'ndf' : 64,
        'netD' : 'n_layers', # 'basic', 'n_layers', 'pixel'
        'netG' : 'resnet_9blocks', # 'unet_256', 'unet_128', 'resnet_6blocks', 'resnet_9blocks'
        'n_layers_D' : 4, # -------------- CHANGED FROM THE USUAL 5 --------------
        'norm' : 'batch', # 'batch', 'instance', 'none'
        'init_type' : 'xavier', # 'normal', 'xavier', 'kaiming', 'orthogonal'
        'init_gain' : 0.02,
        'no_dropout' : False,
        'serial_batches' : False,
        'num_threads' : 4,
        'batch_size' : 1,
        'max_dataset_size' : 17000, # Something like this
        'display_freq' : 2000,
        'print_freq' : 100,
        'valid_freq' : 8500, # 'epoch' for at the end of each epoch
        'num_valid' : 2000,
        'save_latest_freq' : 10000,
        'save_epoch_freq' : 4,
        'phase' : 'train',
        'n_epochs' : 12,
        'n_epochs_decay' : 7,
        'beta1' : 0.5,
        # 'lamda_L1_reg' : 0.005, # 0 for no L1 regularisation
        'adam_weight_decay' : 0.0001, # 0 is default, 0.001
        'lr' : 0.0001, # 0.0002, 0.00005
        'gan_mode' : 'vanilla', # 'vanilla', 'lsgan', 'wgangp
        'pool_size' : 0,
        'lr_policy' : 'linear', # 'linear', 'step', 'plateau', 'cosine'
        'lr_decay_iters' : 50,
        'isTrain' : True,
        'lambda_pix' : 1000, # 1000
        'nonzero_L1weight': 10, # used for none_weighted mask type
        'lambda_channel' : 2, # 20
        'G_output_layer' : 'tanh+clampinduction', # 'identity', 'tanh', 'linear', 'relu', 'tanh+clampcollection', 'tanh+clampinduction'
        'direction' : 'AtoB',
        'channel_offset' : 0, # Induction 112, collection 16
        'tick_offset' : 0, # 58 NOTE both channel and tick offsets need to be nonzero for either of them to be applied
        'unconditional_D' : False, # need True if we want to load into a cycleGAN setup
        'noise_layer' : False,
        'kernel_size' : (3,5), # 3, 4, (3,5)
        'outer_stride' : 2, # 2, (1,3)
        'inner_stride_1' : (1,3) # 2, (1,3)
    }
    # epoch : 'latest' for test

    # If data is not on the current node, grab it from the share disk.
    if not os.path.exists(options['dataroot']):
        options['dataroot'] = options['dataroot_shared_disk']

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
