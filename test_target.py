import os
import torch
from utils.Dataset_seg2D import make_test_loader
from Models.attention_unet import AttU_Net, AttU_Net_TF
from Dataset_target import get_data
import numpy as np


def test(index_path, model, data_modes, roi_modes, input_shape, save_folder=None):
    index_list = np.load(index_path).tolist()
    for case in index_list:
        pred = []
        print("case_name:", case)
        test_loader = make_test_loader(data_root, data_modes, roi_modes, case, input_shape, get_data, batch_size=1)
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.cuda(0)
                output = model(batch_x)[0]
                output = torch.nn.functional.softmax(output, dim=1)
                seg_slice = output.cpu().numpy()
                seg_slice = np.squeeze(seg_slice)
                seg_slice = np.transpose(seg_slice, [1, 2, 0])
                pred.append(seg_slice)
        pred = np.asarray(pred, dtype=np.float32)
        pred = np.transpose(pred, [1, 2, 0, 3])

        if save_folder is not None:
            save_path = os.path.join(save_folder, case)
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path, 'pred_RN.npy'), pred[..., 1])
            np.save(os.path.join(save_path, 'pred_SN.npy'), pred[..., 2])
            np.save(os.path.join(save_path, 'pred_STN.npy'), pred[..., 3])


if __name__ == '__main__':
    model_path = r'./trained_models/fode0_model.pth'
    # model_path = r'./trained_models/fode1_model.pth'
    # model_path = r'./trained_models/fode2_model.pth'
    # model_path = r'./trained_models/fode3_model.pth'
    # model_path = r'./trained_models/fode4_model.pth'
    save_folder = r'./save_folder/fode0'
    data_root = r'./data_folder'
    index_path = r'./index/test_index.npy'
    data_modes = ['qsm.npy']
    roi_modes = ['ROI.npy']
    input_shape = (96, 96)

    in_ch = 3
    n_classes = 4

    model = AttU_Net(img_ch=in_ch, output_ch=n_classes)
    model.load_state_dict(torch.load(model_path)['model'])
    model = model.cuda(0)
    model.eval()
    test(index_path, model,  data_modes, roi_modes, input_shape, save_folder=save_folder)










