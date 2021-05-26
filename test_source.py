import numpy as np
import os
import torch
import SimpleITK as sitk
from utils.Dataset_seg2D import make_test_loader
from Models.attention_unet import AttU_Net
from Dataset_source import get_data



def save_results(seg_volumes, roi_modes, case, store_path):
    folders = os.listdir(path)
    for folder in folders:
        folder_path = os.path.join(path, folder)
        files = os.listdir(folder_path)
        roi_file = [x for x in files if 'ROI.nii' in x]
    seg_list = []
    seg_volumes = np.asarray(seg_volumes)
    pad = np.zeros((1, seg_volumes.shape[-2], seg_volumes.shape[-1]))
    for i in range(len(roi_modes)):
        seg_volume = seg_volumes[:, i, ...]
        seg_result = np.concatenate((pad, seg_volume, pad), axis=0)
        seg_list.append(seg_result)
        if store_path != None:
            save_case_folder = os.path.join(store_path, case)
            os.makedirs(save_case_folder, exist_ok=True)
            image = sitk.GetImageFromArray(seg_result)
            sitk.WriteImage(image, os.path.join(save_case_folder, roi_modes[i] + '.nii.gz'))


def test(index_path, model, data_modes, roi_modes, input_shape, save_folder=None):
    index_list = np.load(index_path).tolist()
    for case in index_list:
        pred = []
        print("case_name:", case)
        test_loader = make_test_loader(data_root, data_modes, roi_modes, case, input_shape, get_data, batch_size=1)
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                # x = batch_x.numpy()
                # plt.imshow(x[0, 0, ...], cmap='gray')
                # plt.show()
                batch_x = batch_x.cuda(0)
                output = model(batch_x)[0]
                output = torch.nn.functional.softmax(output, dim=1)
                seg_slice = output.cpu().numpy()
                seg_slice = np.squeeze(seg_slice)
                seg_slice = np.transpose(seg_slice, [1, 2, 0])
                pred.append(seg_slice)
        pred = np.asarray(pred, dtype=np.float32)
        pred = np.transpose(pred, [1, 2, 0, 3])
        # print(pred.shape)
        # Imshow3DArray(pred[..., 0])
        # Imshow3DArray(pred[..., 1])
        # Imshow3DArray(pred[..., 2])

        if save_folder is not None:
            save_path = os.path.join(save_folder, case)
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path, 'pred_RN.npy'), pred[..., 1])
            np.save(os.path.join(save_path, 'pred_SN.npy'), pred[..., 2])


if __name__ == '__main__':
    model_path = r'/home/wyd/PycharmProjects/ZWW_seg/log1/model/Epoch499_best_model.pth'
    save_folder = r'/home/wyd/PycharmProjects/ZWW_seg/data/seg_low_focal'
    data_root = r'/home/wyd/PycharmProjects/ZWW_seg/data/Low'
    index_path = r'/home/wyd/PycharmProjects/ZWW_seg/data/Low_index/index/test_index.npy'
    data_modes = ['qsm.npy']
    roi_modes = ['ROI.npy']
    input_shape = (128, 128)

    in_ch = 3
    n_classes = 3

    model = AttU_Net(img_ch=in_ch, output_ch=n_classes)
    model.load_state_dict(torch.load(model_path)['model'])
    model = model.cuda(0)
    model.eval()
    test(index_path, model,  data_modes, roi_modes, input_shape, save_folder=save_folder)










