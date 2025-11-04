import sys
import time
import torch
import argparse
from tqdm import tqdm
import random
from Val import val
from Utils import *
from Model import Network
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--n_steps', type=int, default=10, help='number of epochs to update learning rate')
    parser.add_argument('--n_epochs', type=int, default=25, help='number of epochs to train')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decaying factor')
    parser.add_argument('--seed', default=42, type=int, metavar='N', help='Seed for initializing training')
    parser.add_argument('--trainset_dir', type=str, default='./Datasets/WLFI_224x224_dual')

    return parser.parse_args()


def train(train_loader, cfg, test_scene_id):
    os.makedirs('./log/' + str(test_scene_id[0]) + '_' + str(test_scene_id[-1]))
    sys.stdout = open('./log/' + str(test_scene_id[0]) + '_' + str(test_scene_id[-1])
                      + '/' + str(test_scene_id[0]) + '_' + str(test_scene_id[-1]) + '.txt', 'a')
    print(cfg)
    print(test_scene_id)

    net = Network().to(cfg.device)
    cudnn.benchmark = True
    optimizer = torch.optim.SGD([paras for paras in net.parameters() if paras.requires_grad == True],
                                lr=cfg.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)

    best_SRCC = 0
    for idx_epoch in range(0, cfg.n_epochs):

        loss_epoch = []
        loss_list = []
        start_time = time.time()

        for idx_iter, (data_left_h, data_left_v, data_right_h, data_right_v, score_label, class_label) \
                in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
            data_left_h = Variable(data_left_h).to(cfg.device)
            data_left_v = Variable(data_left_v).to(cfg.device)
            data_right_h = Variable(data_right_h).to(cfg.device)
            data_right_v = Variable(data_right_v).to(cfg.device)
            score_label = Variable(score_label).to(cfg.device)
            score_label = score_label.view(score_label.size()[0], -1)
            class_label = Variable(class_label).to(cfg.device)
            class_label = class_label.view(class_label.size()[0])

            score_out, class_out = net(data_left_h, data_left_v, data_right_h, data_right_v)
            loss_main = torch.nn.MSELoss().to(cfg.device)(score_out, score_label)
            loss_aux = torch.nn.CrossEntropyLoss().to(cfg.device)(class_out, class_label)
            loss = loss_main + 0.3 * loss_aux

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.data.cpu())

        loss_list.append(float(np.array(loss_epoch).mean()))
        end_time = time.time()
        print('Test Epoch----%5d,'
              ' loss---%f,'
              ' Time---%f s'
              ' lr---%7f s'
              % (idx_epoch + 1,
                 float(np.array(loss_epoch).mean()),
                 end_time - start_time,
                 scheduler.get_last_lr()[0]))

        save_ckpt({'epoch': idx_epoch + 1, 'state_dict': net.state_dict(), 'loss': loss_list, },
                  save_path='./log/' + str(test_scene_id[0]) + '_' + str(test_scene_id[-1]) + '/',
                  filename='MBDV-BWLFQ_epoch' + str(idx_epoch + 1) + '.pth.tar')
        load_model_path = './log/' + str(test_scene_id[0]) + '_' + str(test_scene_id[-1]) + \
                          '/MBDV-BWLFQ_epoch' + str(idx_epoch + 1) + '.pth.tar'

        start_time_val = time.time()
        val_SRCC = val(valset_dir=cfg.trainset_dir, test_scene_id=test_scene_id, load_model_path=load_model_path)
        end_time_val = time.time()
        print('Val_Time----    %f s'
              % (end_time_val - start_time_val)
              )

        if val_SRCC > best_SRCC:
            best_SRCC = val_SRCC
            save_ckpt({'state_dict': net.state_dict()},
                      save_path='./log/' + str(test_scene_id[0]) + '_' + str(test_scene_id[-1]) + '/', filename='best_model.pth.tar')

        if (idx_epoch + 1) != cfg.n_epochs:
            os.system('rm -r ./log/' + str(test_scene_id[0]) + '_' + str(test_scene_id[-1]) + '/'
                      + 'MBDV-BWLFQ_epoch' + str(idx_epoch + 1) + '.pth.tar')

        scheduler.step()


def save_ckpt(state, save_path='./log', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path, filename))


def main(cfg):
    scene_num = 40
    index = list(range(0, scene_num))
    num_groups = 5
    group_size = len(index) // num_groups
    scene_groups = [index[i * group_size: (i + 1) * group_size] for i in range(num_groups)]
    full_dataset_dir = cfg.trainset_dir
    for test_scene_id in scene_groups:
        train_set = MyTrainSetLoader_Kfold(dataset_dir=full_dataset_dir, test_scene_id=test_scene_id)
        train_loader = DataLoader(dataset=train_set, num_workers=cfg.num_workers, batch_size=cfg.batch_size, shuffle=True)
        train(train_loader, cfg, test_scene_id)


if __name__ == '__main__':
    cfg = parse_args()

    # set random seed
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    main(cfg)
