import numpy as np
import os


def esemble_result(folder, key_name_list, is_save=False):
    fode_files = [x for x in os.listdir(folder) if  'fode' in x]
    names = os.listdir(os.path.join(folder, fode_files[0]))
    for key_name in key_name_list:
        for name in names:
            data_list = []
            for fode in fode_files:
                file = os.path.join(folder, fode, name, key_name)
                data = np.load(file)
                data_list.append(data)
            data_array = np.asarray(data_list, dtype=np.float32)
            data_array = np.mean(data_array, axis=0)
            print(data_array.shape)
            if is_save:
                save_folder = os.path.join(folder, 'esemble_result', name)
                os.makedirs(save_folder, exist_ok=True)
                np.save(os.path.join(save_folder, key_name), data_array)


if __name__ == '__main__':
    folder_path = r'./save_folder'
    key_name = ['pred_RN.npy', 'pred_SN.npy', 'pred_STN.npy']
    # key_name = ['pred_RN.npy', 'pred_SN.npy']
    esemble_result(folder_path, key_name, is_save=True)


