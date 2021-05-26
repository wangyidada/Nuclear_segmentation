import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.my_utils import getDataList, get_test_data_list
import albumentations
import cv2


class Datasets2D(Dataset):
    def __init__(self, data_folder, case_list, data_modes, roi_modes, input_shape, get_data,
                 is_training=True, is_normalize=True):
        self.data_folder = data_folder
        self.case_list = case_list
        self.data_modes = data_modes
        self.roi_modes = roi_modes
        self.resize_shape = input_shape
        self.is_training = is_training
        self.is_normlize = is_normalize
        self.get_data = get_data

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, index):
        case = self.case_list[index]
        # the shape of data_volumes: [[H, W, 3], [H, W, 3], ......]
        # the shape of seg_volumes: [[H, W, 1], [H, W, 1], [H, W, 1]]

        data_volumes, seg_volumes = self.get_data(self.data_folder, case, self.data_modes, self.roi_modes, is_3slice=True)
        n, c = len(data_volumes), np.shape(data_volumes)[-1]
        # the shape of data_volumes: [H, W, C]
        # the shape of seg_volumes: [H, W, S]
        data_volumes = np.concatenate(data_volumes, axis=-1).astype(np.float32)
        seg_volumes = np.concatenate(seg_volumes, axis=-1).astype(np.float32)
        data_volumes, seg_volumes = self.aug_sample(data_volumes, seg_volumes, is_training=self.is_training)
        if self.is_normlize ==True:
            data_list = [data_volumes[..., c*x: c*(x + 1)] for x in range(n)]
            normlized_data = [self.normlize(x) for x in data_list]
            data_volumes = np.concatenate(normlized_data, axis=-1).astype("float32")
        # data_input:[channel*c, h, w]
        # mask_input: [channel, h, w]
        data_input = np.transpose(data_volumes, [2, 0, 1]).astype('float32')
        mask_input = np.transpose(seg_volumes, [2, 0, 1]).astype('float32')
        return (torch.tensor(data_input.copy(), dtype=torch.float),
                torch.tensor(mask_input.copy(), dtype=torch.float))

    def aug_sample(self, image, mask, is_training=True):
        if is_training:
            train_tranform = albumentations.Compose([
                albumentations.Resize(self.resize_shape[0], self.resize_shape[1]),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.VerticalFlip(p=0.5),
                albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10,
                                                interpolation=cv2.INTER_LINEAR, p=0.9),
                # albumentations.ElasticTransform(p=0.5, alpha=1, sigma=50, alpha_affine=5)
            ])
        else:
            train_tranform = albumentations.Compose([
                albumentations.Resize(self.resize_shape[0], self.resize_shape[1])])
        transformed = train_tranform(image=image, mask=mask)
        aug_image = transformed['image']
        aug_mask = transformed['mask']
        # from MeDIT.Visualization import Imshow3DArray
        # Imshow3DArray(aug_image)
        # Imshow3DArray(aug_mask)
        return aug_image, aug_mask

    def normlize(self, x):
        # x = x / 100
        # x[x < -10] = 0
        # return x
        return (x - x.mean()) / (x.std() + 1e-7)


def make_data_loaders(data_root, data_modes, roi_modes, index_path, input_shape, get_data, batch_size=8):
    train_list = getDataList(os.path.join(index_path, 'train_index.npy'), rate=0.3)
    val_list = getDataList(os.path.join(index_path, 'val_index.npy'), rate=0.3)

    train_ds = Datasets2D(data_root, train_list, data_modes, roi_modes, input_shape, get_data)
    val_ds = Datasets2D(data_root, val_list, data_modes, roi_modes, input_shape, get_data)
    loaders = {}
    loaders['train'] = DataLoader(train_ds, batch_size=batch_size,
                              num_workers=0, shuffle=True)
    loaders['eval'] = DataLoader(val_ds, batch_size=batch_size,
                             num_workers=0, shuffle=True)
    return loaders


def make_test_loader(data_root, data_modes, roi_modes, case, input_shape,  get_data, batch_size=1):
    test_list = get_test_data_list(data_root, case, data_modes, is_3slice=True)
    test_ds = Datasets2D(data_root, test_list, data_modes, roi_modes, input_shape, get_data, is_training=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                                 num_workers=0, shuffle=False)
    return test_loader




