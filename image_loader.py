import os
import skimage.io
from imageio import imread
import cv2
import random
import numpy as np
import torch
import utils

cv2.setNumThreads(0)


# type is a string that appeared in the image file name, should be chosen from 'color' or 'depth'
def get_image_filenames(dir, keyword=None):
    """Returns all files in the input directory dir that are images"""
    image_types = ('jpg', 'jpeg', 'tiff', 'tif', 'png', 'bmp', 'gif', 'exr', 'dpt', 'hdf5','webp','pfm', 'pt')
    if isinstance(dir, str):
        files = os.listdir(dir)
        exts = (os.path.splitext(f)[1] for f in files)
        if keyword != None:
            images = [os.path.join(dir, f)
                        for e, f in zip(exts, files)
                        if e[1:] in image_types and keyword in f]
        else:
            images = [os.path.join(dir, f)
                        for e, f in zip(exts, files)
                        if e[1:] in image_types]
        return images
    elif isinstance(dir, list):
        # Suppport multiple directories (randomly shuffle all)
        images = []
        for folder in dir:
            files = os.listdir(folder)
            exts = (os.path.splitext(f)[1] for f in files)
            if keyword != None:
                images_in_folder = [os.path.join(folder, f)
                                    for e, f in zip(exts, files)
                                    if e[1:] in image_types and keyword in f]
            else:
                images_in_folder = [os.path.join(folder, f)
                                    for e, f in zip(exts, files)
                                    if e[1:] in image_types]
            images = [*images, *images_in_folder]
        return images



class PairsLoader(torch.utils.data.IterableDataset):
    """Loads (image, masks, phase) tuples for reverse model training

    Class initialization parameters
    -------------------------------

    :param data_path:
    :param plane_idxs:
    :param batch_size:
    :param image_res:
    :param shuffle:

    """

    def __init__(self, data_path, plane_idxs=None,
                 image_res=(800, 1280), shuffle=True,
                ):
        """

        """
        print(data_path)
        if isinstance(data_path, str):
            if not os.path.isdir(data_path):
                raise NotADirectoryError(f'Data folder: {data_path}')
            self.image_path = os.path.join(data_path, 'image')
            self.mask_path = os.path.join(data_path, 'mask')
            self.phase_path = os.path.join(data_path, 'phase')
        elif isinstance(data_path, list):
            self.image_path = [os.path.join(path, 'image') for path in data_path]
            self.mask_path = [os.path.join(path, 'mask') for path in data_path]
            self.phase_path = [os.path.join(path, 'phase') for path in data_path]
        

        self.all_plane_idxs = plane_idxs
        self.shuffle = shuffle
        self.image_res = image_res
        
        
        self.image_names = get_image_filenames(self.image_path)
        self.image_names.sort()
        self.mask_names = get_image_filenames(self.mask_path)
        self.mask_names.sort()
        self.phase_names = get_image_filenames(self.phase_path)
        self.phase_names.sort()

        # create list of image IDs with augmentation state
        self.order = [i for i in range(len(self.image_names))]

    def __iter__(self):
        self.ind = 0
        if self.shuffle:
            random.shuffle(self.order)
        return self

    def __len__(self):
        return len(self.image_names)

    def __next__(self):
        if self.ind < len(self.order):
            pair_idx = self.order[self.ind]

            self.ind += 1
            return self.load_pair(pair_idx)
        else:
            raise StopIteration

    def load_pair(self, filenum):
        image_path = self.image_names[filenum]
        mask_path = self.mask_names[filenum]
        phase_path = self.phase_names[filenum]

        # load image, mask, phase
        # image = torch.load(image_path)
        # mask = torch.load(mask_path)
        # phase = torch.load(phase_path)
        
        ret = [torch.load(image_path), torch.load(mask_path), torch.load(phase_path)] # image, mask, phase
        for i, target in enumerate(ret):
            if type(target) != torch.tensor:
                target = torch.tensor(target)
            ret[i] = self.reshape_tensor_to_3dim(target)

        return tuple(ret)
    
    def reshape_tensor_to_3dim(self, tensor):
        len_dim = len(tensor.shape)
        if len_dim > 3:
            tensor = tensor.squeeze()
        
        while len(tensor.shape) < 3:
            tensor = tensor.unsqueeze(0)
        
        return tensor

# def reshape_to_3d(tensor):
#         len_dim = len(tensor.shape)
#         if len_dim > 3:
#             tensor = tensor.squeeze()
        
#         while len(tensor.shape) < 3:
#             tensor = tensor.unsqueeze(0)
        
#         return tensor

# if __name__ == '__main__':
#     tensor_5d = torch.randn(1, 1, 1, 480, 640)
#     tensor_5d_3 = torch.randn(1, 1, 3, 480, 640)
#     tensor_2d = torch.randn(480, 640)

#     tensor_3d_5d = reshape_to_3d(tensor_5d)
#     tensor_3d_5d_3 = reshape_to_3d(tensor_5d_3)
#     tensor_3d_2d = reshape_to_3d(tensor_2d)

#     print(tensor_3d_5d.shape)  # torch.Size([1, 480, 640])
#     print(tensor_3d_5d_3.shape)  # torch.Size([1, 480, 640])
#     print(tensor_3d_2d.shape)  # torch.Size([1, 480, 640])




