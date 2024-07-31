import os, random, time

import numpy as np
import torch
import sparse

from matplotlib import pyplot as plt

class Dataset():
    """
    A dataset class for paired image dataset. Either aligned or unaligned.
    """
    def __init__(self, opt, valid=False, nd_ped=False):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

        if not valid:
            self.dir_A = os.path.join(opt.dataroot, "trainA")
            self.dir_B = os.path.join(opt.dataroot, "trainB")

        else:
            self.dir_A = os.path.join(opt.dataroot, 'validA')
            self.dir_B = os.path.join(opt.dataroot, 'validB')

        self.A_paths = sorted(self._make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(self._make_dataset(self.dir_B, opt.max_dataset_size))

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
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        A = sparse.load_npz(A_path).todense() if self.opt.nd_sparse else np.load(A_path)
        B = np.load(B_path).astype(float) # NOTE not necessary if make_infilled_pairs is run again

        if self.opt.crop_w or self.opt.crop_h:
            A = A[:, self.opt.crop_h:-self.opt.crop_h, self.opt.crop_w:-self.opt.crop_w]
            B = B[:, self.opt.crop_h:-self.opt.crop_h, self.opt.crop_w:-self.opt.crop_w]

        elif self.opt.pad_w or self.opt.pad_h:
            B_c, B_h, B_w = B.shape
            padded_B = np.zeros((B_c, B_h + (2 * self.opt.pad_h), B_w + (2 * self.opt.pad_w)))
            padded_B[:, self.opt.pad_h:-self.opt.pad_h, self.opt.pad_w:-self.opt.pad_w] = B
            B = padded_B
            A_c, A_h, A_w = A.shape
            padded_A = np.zeros((A_c, A_h + (2 * self.opt.pad_h), A_w + (2 * self.opt.pad_w)))
            padded_A[:, self.opt.pad_h:-self.opt.pad_h, self.opt.pad_w:-self.opt.pad_w] = A
            A = padded_A

        if self.opt.mask_type in ['saved', 'saved_1rms']:
            full_mask = A[-1:, :, :]
            A = A[:-1, :, :]

        elif self.opt.mask_type == 'saved_fd':
            full_mask = B[-1:, :, :]
            B = B[:-1, :, :]

        else:
            # Can't be bothered to refactor out the mask object when not in the data so just
            # throw an empty one around
            full_mask = np.zeros((1, B.shape[1], B.shape[2]))
            if self.opt.mask_type == 'dont_use':
              A = A[:-1, :, :]

        input_nc = self.opt.input_nc - 1 if self.opt.noise_layer else self.opt.input_nc

        for i in range(self.opt.output_nc):
            B[i] *= self.opt.B_ch_scalefactors[i]
        for i in range(input_nc):
            A[i] *= self.opt.A_ch_scalefactors[i]

        if self.opt.samples == -1: # Don't take crops of the image
            A = torch.from_numpy(A).float()
            B = torch.from_numpy(B).float()
            mask = torch.from_numpy(full_mask[:, :, :]).float()

        else:
            raise NotImplementedError("I don't think opt.samples != -1 works anymore, check!")

            # A_tiles, B_tiles = [], []
            # samples = self.opt.samples if self.opt.samples else A.shape[2] // 512
            # mask = []
            # # Want to get a 512x512 crop of image (collection view images are saved as 512x4492).
            # for _ in range(samples):
            #     bad_tile = True
            #     while bad_tile:
            #         tick = random.randint(0, A.shape[2] - 512)

            #         A_tile = A[:, :, tick:tick + 512]
            #         B_tile = B[:, :, tick:tick + 512]

            #         if A_tile[0].sum() != 0:
            #             bad_tile = False

            #     A_tiles.append(torch.from_numpy(A_tile).float())
            #     B_tiles.append(torch.from_numpy(B_tile).float())
            #     mask.append(torch.from_numpy(full_mask[:, :, tick:tick + 512]).float())

            # A = A_tiles
            # B = B_tiles

        if self.opt.noise_layer:
            # noise RMS is ~4 so use 4/4096 as sigma for Gaussian noise channel
            noise = torch.normal(0, 0.0009765625, size=(1, A.size()[1], A.size()[2]))
            A = torch.cat((A, noise), 0)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'mask' : mask}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)

    def _make_dataset(self, dir, max_dataset_size=float("inf")):
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
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
        self.dataset = Dataset(opt, valid, nd_ped)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        if not valid:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=not opt.serial_batches,
                num_workers=int(opt.num_threads)
            )
        else:
             self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0
            )

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
