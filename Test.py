import torch
from Utils import *
from Model import Network
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from scipy.stats import spearmanr as SROCC


def test_model():
    # WLFI_NBU
    load_all_model_path = 'PreTrainedModels/'
    valset_dir = './Datasets/WLFI_224x224_dual/'
    dataset_name = 'WLFI_NBU'
    scene_list = ['Scene_1', 'Scene_2', 'Scene_3', 'Scene_4', 'Scene_5', 'Scene_6', 'Scene_7', 'Scene_8', 'Scene_9', 'Scene_10',
                  'Scene_11', 'Scene_12', 'Scene_13', 'Scene_14', 'Scene_15', 'Scene_16', 'Scene_17', 'Scene_18', 'Scene_19', 'Scene_20',
                  'Scene_21', 'Scene_22', 'Scene_23', 'Scene_24', 'Scene_25', 'Scene_26', 'Scene_27', 'Scene_28', 'Scene_29', 'Scene_30',
                  'Scene_31', 'Scene_32', 'Scene_33', 'Scene_34', 'Scene_35', 'Scene_36', 'Scene_37', 'Scene_38', 'Scene_39', 'Scene_40']
    test_scene_num = 8
    distorted_num = 8
    scene_num = 40

    device = 'cuda:0'
    net = Network().to(device)
    cudnn.benchmark = True

    all_model = os.listdir(load_all_model_path)
    label_list = np.zeros([test_scene_num * distorted_num, len(all_model)])
    data_list = np.zeros([test_scene_num * distorted_num, len(all_model)])
    val_SRCC_all = []
    test_scene_id_list = []

    index = list(range(0, scene_num))
    num_groups = 5
    group_size = len(index) // num_groups
    scene_groups = [index[i * group_size: (i + 1) * group_size] for i in range(num_groups)]
    for test_scene_id in scene_groups:
        test_scene_id_list.append(test_scene_id)

    for id, model_name in enumerate(test_scene_id_list):
        load_model_path = load_all_model_path + '/' + str(model_name[0]) + '_' + str(model_name[-1]) + '/best_model.pth.tar'
        model = torch.load(load_model_path, map_location={'cuda:0': device})
        net.load_state_dict(model['state_dict'])
        net.eval()
        index = 0
        test_scene_id = model_name
        for test_scene in test_scene_id:
            image_path = valset_dir + '/' + scene_list[test_scene]
            image_list = os.listdir(image_path)
            for test_image in image_list:
                patch_path = valset_dir + '/' + scene_list[test_scene] + '/' + test_image
                patch_list = os.listdir(patch_path)
                output_list = 0
                for val_patch in patch_list:
                    each_patch_path = patch_path + '/' + val_patch
                    with h5py.File(each_patch_path, 'r') as hf:
                        label = np.array(hf.get('score_label'))

                        data_left_h = np.array(hf.get('data_left_h'))
                        data_left_h = data_left_h / 255
                        data_left_h = np.expand_dims(data_left_h, axis=0)
                        data_left_h = torch.from_numpy(data_left_h.copy())
                        data_left_h = Variable(data_left_h).to(device)

                        data_left_v = np.array(hf.get('data_left_v'))
                        data_left_v = data_left_v / 255
                        data_left_v = np.expand_dims(data_left_v, axis=0)
                        data_left_v = torch.from_numpy(data_left_v.copy())
                        data_left_v = Variable(data_left_v).to(device)

                        data_right_h = np.array(hf.get('data_right_h'))
                        data_right_h = data_right_h / 255
                        data_right_h = np.expand_dims(data_right_h, axis=0)
                        data_right_h = torch.from_numpy(data_right_h.copy())
                        data_right_h = Variable(data_right_h).to(device)

                        data_right_v = np.array(hf.get('data_right_v'))
                        data_right_v = data_right_v / 255
                        data_right_v = np.expand_dims(data_right_v, axis=0)
                        data_right_v = torch.from_numpy(data_right_v.copy())
                        data_right_v = Variable(data_right_v).to(device)

                    with torch.no_grad():
                        out_score, _ = net(data_left_h, data_left_v, data_right_h, data_right_v)
                    output_list += out_score.cpu().numpy()
                label_list[index, id] = label.item()
                data_list[index, id] = output_list.item() / len(patch_list)
                index += 1

        val_SRCC = SROCC(data_list[:, id], label_list[:, id]).correlation
        val_SRCC_all.append(val_SRCC)
        print(test_scene_id)
        print('SROCC :----    %f' % val_SRCC)
    print('Average SROCC :----   %f' % np.mean(val_SRCC_all))

    # save in h5 file and test in matlab
    f = h5py.File('./Results/MBDV-BWLFQ_result_' + dataset_name + '.h5', 'w')
    f.create_dataset('predict_data', data=data_list)
    f.create_dataset('score_label', data=label_list)
    f.close()


if __name__ == '__main__':
    test_model()
