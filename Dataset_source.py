import os
import numpy as np
from utils.Dataset_seg2D import make_data_loaders, make_test_loader
from utils.my_utils import load_data


def get_data(path, each_case, data_modes, roi_modes, is_3slice=True):
    # print(each_case)
    folder_files = os.listdir(os.path.join(path, each_case[0]))
    s = each_case[1]
    data_list = []
    roi_list = []
    for data_mode in data_modes:
        data_file = [x for x in folder_files if data_mode in x][0]
        data_path = os.path.join(path, each_case[0], data_file)
        if is_3slice:
            data = load_data(data_path)
            data_cut = data[s-1:s+2, ...]
            data_cut = np.transpose(data_cut, [1, 2, 0])
        else:
            data = load_data(data_path)
            data_cut = data[s:s+1, ...]
            data_cut = np.transpose(data_cut, [1, 2, 0])
        data_list.append(data_cut)

    for roi_mode in roi_modes:
        roi_file = [x for x in folder_files if roi_mode in x][0]
        roi_path = os.path.join(path, each_case[0], roi_file)
        roi_data = load_data(roi_path)
        roi_cut = roi_data[s:s + 1, ...]
        roi_cut = np.transpose(roi_cut, [1, 2, 0])
        roi_list.append(roi_cut)
    return data_list, roi_list


def get_test_data_list(folder_path, case, data_mode, is_3slice=True):
    data_list = []
    case_files = os.listdir(os.path.join(folder_path, case))
    data_name = [x for x in case_files if data_mode[0] in x]
    assert len(data_name) <= 1, "wong roi_mode"
    data_path = os.path.join(folder_path, case, data_name[0])
    data = load_data(data_path)
    s = data.shape[0]
    if is_3slice:
        for i in range(1, s - 1):
            data_list.append((case, i))
    else:
        for i in range(s):
            data_list.append((case, i))
    return data_list



if __name__ == "__main__":
    data_root = r'/home/wyd/PycharmProjects/ZWW_seg/data/Low'
    dict_path = r'/home/wyd/PycharmProjects/ZWW_seg/data/Low_index/dict_index'
    data_modes = ['qsm.npy']
    roi_modes = ['ROI.npy']
    input_shape = (128, 128)

    datasets = make_data_loaders(data_root, data_modes, roi_modes, dict_path, input_shape, get_data, batch_size=1)
    # datasets = make_test_loader(data_root, data_modes, roi_modes, case, input_shape, get_data, batch_size=1)
    # for x, y in datasets:
    for x, y in datasets['train']:
        images_data = x.numpy()
        seg_data = y.numpy()
        print(images_data.shape, seg_data.shape)
        for i in range(1):
            from MeDIT.Visualization import Imshow3DArray
            x = np.transpose(images_data[0, ...])
            Imshow3DArray(np.transpose(images_data[0, ...], [1, 2, 0]))
            print(np.max(images_data), np.unique(seg_data))
            import matplotlib.pyplot as plt
            plt.imshow(images_data[i, 1, ...], cmap='gray')
            plt.contour(seg_data[i, 0, ...])
            plt.show()




