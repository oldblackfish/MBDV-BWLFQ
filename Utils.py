import os
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import numpy as np
import h5py


class MyTrainSetLoader_Kfold(Dataset):
    def __init__(self, dataset_dir, test_scene_id):
        super(MyTrainSetLoader_Kfold, self).__init__()
        self.dataset_dir = dataset_dir
        scene_list = ['Scene_1', 'Scene_2', 'Scene_3', 'Scene_4', 'Scene_5', 'Scene_6', 'Scene_7', 'Scene_8', 'Scene_9', 'Scene_10',
                      'Scene_11', 'Scene_12', 'Scene_13', 'Scene_14', 'Scene_15', 'Scene_16', 'Scene_17', 'Scene_18', 'Scene_19', 'Scene_20',
                      'Scene_21', 'Scene_22', 'Scene_23', 'Scene_24', 'Scene_25', 'Scene_26', 'Scene_27', 'Scene_28', 'Scene_29', 'Scene_30',
                      'Scene_31', 'Scene_32', 'Scene_33', 'Scene_34', 'Scene_35', 'Scene_36', 'Scene_37', 'Scene_38', 'Scene_39', 'Scene_40']

        for _ in test_scene_id:
            scene_list.pop(test_scene_id[0])

        all_patch_path = []
        for scene in scene_list:
            distorted_scene_list = os.listdir(dataset_dir + '/' + scene)
            for distorted_scene in distorted_scene_list:
                distorted_path_list = os.listdir(dataset_dir + '/' + scene + '/' + distorted_scene)
                for distorted_path in distorted_path_list:
                    path = scene + '/' + distorted_scene + '/' + distorted_path
                    all_patch_path.append(path)

        self.all_patch_path = all_patch_path
        self.item_num = len(self.all_patch_path)

    def __getitem__(self, index):
        all_patch_path = self.all_patch_path
        dataset_dir = self.dataset_dir
        file_name = dataset_dir + '/' + all_patch_path[index]
        with h5py.File(file_name, 'r') as hf:
            data_left_h = np.array(hf.get('data_left_h'))
            data_left_h = data_left_h / 255
            data_left_h = np.transpose(data_left_h, [1, 2, 0])
            data_left_h = ToTensor()(data_left_h.copy())

            data_left_v = np.array(hf.get('data_left_v'))
            data_left_v = data_left_v / 255
            data_left_v = np.transpose(data_left_v, [1, 2, 0])
            data_left_v = ToTensor()(data_left_v.copy())

            data_right_h = np.array(hf.get('data_right_h'))
            data_right_h = data_right_h / 255
            data_right_h = np.transpose(data_right_h, [1, 2, 0])
            data_right_h = ToTensor()(data_right_h.copy())

            data_right_v = np.array(hf.get('data_right_v'))
            data_right_v = data_right_v / 255
            data_right_v = np.transpose(data_right_v, [1, 2, 0])
            data_right_v = ToTensor()(data_right_v.copy())

            score_label = np.array(hf.get('score_label'))
            score_label = ToTensor()(score_label.copy())

            class_label = np.array(hf.get('class')) - 1
            class_label = torch.tensor(class_label, dtype=torch.long)
        return data_left_h, data_left_v, data_right_h, data_right_v, score_label, class_label

    def __len__(self):
        return self.item_num
