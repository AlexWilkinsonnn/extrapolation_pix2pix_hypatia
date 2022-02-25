import os, random, time
from PIL import Image
import numpy as np

#import torchvision.transforms as transforms
import torch

from matplotlib import pyplot as plt

class AlignedDataset():
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """
    def __init__(self, opt, valid=False, nd_ped=False):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

        if not valid:
            self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory

        else:
            self.dir_AB = os.path.join(opt.dataroot, "valid")

        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc

        self.nd_ped = nd_ped

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = np.load(AB_path)

        if False: #(self.opt.input_nc == 1 and self.opt.output_nc == 1):
            AB = np.expand_dims(AB, axis=0) # Add a channel dimension

        # AB = Image.open(AB_path)
        # split AB image into A and B
        _, h, w = AB.shape
        w2 = int(w / 2)
        if self.opt.direction == 'AtoB':
            A = AB[:,:,:w2]
            B = AB[:,:,w2:]

        elif self.opt.direction == 'BtoA':
            B = AB[:,:,:w2]
            A = AB[:,:,w2:]

        else:
            raise NotImplementedError('direction %s is not valid' % direction)

        # if self.opt.using_mask: # Keeping mask in the unused B channel
        if self.opt.mask_type in ['saved', 'saved_1rms']:
            full_mask = B[1:, :, :]
        else:
            full_mask = np.zeros((1, B.shape[1], B.shape[2])) # Can't be bothered to refactor out the mask object when not in the data so have this.

        # Aligned images need the same channels so if input_nc != output_nc there will be zero channels in one of A and B.
        A = A[:self.opt.input_nc, :, :]
        B = B[:self.opt.output_nc, :, :]

        A[0]*=self.opt.A_ch0_scalefactor
        # A[1]*=self.opt.A_ch1_scalefactor
        B[0]*=self.opt.B_ch0_scalefactor
        
        A_tiles, B_tiles = [], []
        samples = self.opt.samples if self.opt.samples else A.shape[2] // 512
        mask = [] if self.opt.mask_type == 'saved' else [0]*samples
        if self.opt.full_image: # Want to get a 512x512 crop of image (collection view images are saved as 512x4492).
            for tiles in range(samples):
                bad_tile = True
                while bad_tile:
                    tick = random.randint(0, A.shape[2] - 512)

                    A_tile = A[:, :, tick:tick + 512]
                    B_tile = B[:, :, tick:tick + 512]

                    if self.opt.direction == 'AtoB' and A_tile[0].sum() != 0:
                        bad_tile = False
                    elif self.opt.direction == 'BtoA' and B_tile[0].sum() != 0:
                        bad_tile = False

                A_tiles.append(torch.from_numpy(A_tile).float())
                B_tiles.append(torch.from_numpy(B_tile).float())
                if self.opt.mask_type == 'saved':
                    mask.append(torch.from_numpy(full_mask[:, :, tick:tick + 512]).float())

            A = A_tiles
            B = B_tiles

        else:
            A = torch.from_numpy(A).float()
            B = torch.from_numpy(B).float()
            mask = torch.from_numpy(full_mask[:, :, :]).float()
            # if self.opt.using_mask:
            #     mask.append(torch.from_numpy(full_mask[:, :, :]).float())

        # if self.opt.noise_layer:
        #     noise = torch.randn((1,512,512))
        #     A = torch.cat((A, noise), 0)
        
        # Adding back on nd ped for legacy
        if self.nd_ped:
            A[A != 0] += 74.0 
        
        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path, 'mask' : mask}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            # if is_image_file(fname):
            path = os.path.join(root, fname)
            images.append(path)

    return images[:min(max_dataset_size, len(images))]


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""
    def __init__(self, opt, valid=False, nd_ped=False):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        self.dataset = AlignedDataset(opt, valid, nd_ped)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        if not valid:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=not opt.serial_batches,
                num_workers=int(opt.num_threads))
        else:
             self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0)           

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break

            yield data
