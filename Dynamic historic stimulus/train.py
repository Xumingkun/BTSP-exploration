import torch.optim as optim
import numpy as np
import os
import torchvision
import torchvision.transforms as transforms
import argparse
import utils
import torch
import matplotlib.pyplot as plt
import torch.functional as F
from model import *
from tqdm import tqdm
import mat_loader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr_s2', default=2, type=float, help='learning rate')
parser.add_argument('--batch_size', default=batch_size, type=int)
parser.add_argument('--task_permutation_up', default=2312, type=int)
parser.add_argument('--task_permutation_down', default=2312, type=int)
parser.add_argument('--task_times', default=19, type=int)
parser.add_argument('--task_per_group', default=20, type=int)
parser.add_argument('--test_task_num', default=20, type=int)
parser.add_argument('--stage_2_epoch_per_task', default=10, type=int)
parser.add_argument('--inputs_size', default=34*34*2, type=int)
parser.add_argument('--root_path', default='A_4.5', type=str)
parser.add_argument('--seed', default=4300, type=int)
parser.add_argument('--gpu', default=5, type=int)

parser.add_argument('--ID', action="store_true")
parser.add_argument('--XDG', action="store_true")
parser.add_argument('--EWC', action="store_true")

args = parser.parse_args()
setup_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)

if(args.XDG==True):
    gate_thr_xdg = 0.8
else:
    gate_thr_xdg = 0

def gate_manual(task_num):
    gate_list=[]
    for i in range(task_num):
        temp=np.random.rand(cfg_fc[0])
        temp=temp>=gate_thr_xdg
        temp=temp.astype(np.float)
        gate_list.append(temp)
    return torch.FloatTensor(np.array(gate_list)).cuda()

xdg_gate=gate_manual(args.test_task_num)

def create_folders():
    if (os.path.exists(args.root_path)):
        print('alreadly exist')
    else:
        os.mkdir(args.root_path)
        os.mkdir(args.root_path + '/models')

def permute_index(item):
    p_num=int(np.random.rand()*(args.task_permutation_up-args.task_permutation_down))+args.task_permutation_down
    start=int(np.random.rand()*(args.inputs_size-p_num))
    index=item.copy()
    result=index[:start]+ np.random.permutation(index[start:start+p_num]).tolist()+index[start+p_num:]
    return result

def get_permutation_index(times):
    ori_index = [i for i in range(args.inputs_size)]
    ori_index=np.random.permutation(ori_index).tolist()
    index=[ori_index]
    for i in range(times):
        if((i+1)%args.task_per_group==0):
            temp=np.random.permutation(index[-1].copy()).tolist()
        else:
            temp=index[-1]
        index.append(permute_index(temp))
    index = np.array(index)
    print('task num:{}'.format(index.shape[0]))
    np.save(args.root_path + '/p_index.npy', index)
    # similarity_matrix=utils.show_task_similarity(index)
    # np.save(args.root_path + '/p_similarity.npy', similarity_matrix)
    return index

def get_net():
    net = SNN_Model().cuda()
    return net

def get_data_set():
    trainset = mat_loader.dataset_from_mat_dvs('data/npy/train',time_window)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testset = mat_loader.dataset_from_mat_dvs('data/npy/test',time_window)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return trainloader, testloader

def train_snn_task(net, data_set, permutation_index_list):
    test_acc_list = []
    optimizer_main = optim.SGD(net.parameters(), lr=args.lr_s2,momentum=0.9)
    criterion = nn.CrossEntropyLoss().cuda()
    print('training:-------------------')

    for index in range(args.test_task_num):
        for i in range(args.stage_2_epoch_per_task):
            train_snn_epoch(net, i, data_set[0], permutation_index_list, optimizer_main, criterion, index,xdg_gate)

        if(args.EWC==True):
            print('consolidating')
            net.estimate_fisher(data_set[0], xdg_gate[index], args.ID, permutted_paramer=permutation_index_list[index])

        if ((i + 1) % 1 == 0):
            test_acc= test_task(net, data_set[1], permutation_index_list,xdg_gate)
            print('{} accuracy: '.format(i), test_acc)
            test_acc_list.append(test_acc)
            plt.plot(test_acc_list)
            plt.savefig(args.root_path + '/acc.png')
            plt.close()
            utils.save_dict({'acc': test_acc_list}, args.root_path + '/test_acc.pkl')
            torch.save(net.state_dict(), args.root_path + '/models/parameters_{}.pkl'.format(index))

def train_snn_epoch(net, epoch, train_set, permutation_index_list, optimizer_main,criterion,index,xdg_gate):
    net.train()
    def adjust_lr(epoch):
        lr = args.lr_s2 * (0.1 ** (epoch // 5))
        return lr

    lr = adjust_lr(epoch)
    for p in optimizer_main.param_groups:
        p['lr'] = lr
    print('\nEpoch: %d,lr: %.5f' % (epoch, lr))
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_set):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs = inputs[:,:,permutation_index_list[index]]
        optimizer_main.zero_grad()

        pred = net(inputs,xdg_gate[index],args.ID)


        if(args.EWC==True):
            ewc_loss = net.ewc_loss() * net.ewc_lambda
            loss = criterion(pred, targets) + ewc_loss
        else:
            loss = criterion(pred, targets)

        loss.backward()
        optimizer_main.step()
        train_loss += loss.item()
        _, predicted = pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        indicator = int(len(train_set) / 3)

        if ((batch_idx + 1) % indicator == 0):
            print(batch_idx, len(train_set), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

def test_task(net, test_set, permutation_index_list,xdg_gate):
    print('testing-------------------------------------')
    acc_list = []
    for i in tqdm(range(args.test_task_num)):
        acc= test_epoch(net, test_set, permutation_index_list,i,xdg_gate[i])
        acc_list.append(acc)
    print('mean acc:', np.mean(acc_list))
    return acc_list

def test_epoch(net, test_set, permutation_index_list,task_index,xdg_gate):
    net.eval()
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_set):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs = inputs[:,:, permutation_index_list[task_index]]










        pred= net(inputs,xdg_gate,args.ID)








        _, predicted = pred.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    return acc

def print_args(args):
    dict = vars(args)
    print('arguments:--------------------------')
    for key in dict.keys():
        print('{}:{}'.format(key, dict[key]))
    utils.save_dict(dict, args.root_path + '/args.pkl')
    print('-----------------------------------')

def main():
    print(args.root_path)
    create_folders()
    print_args(args)
    print('--------pre_task---------')
    permutation_index_list = get_permutation_index(args.task_times)
    net = get_net()
    data_set = get_data_set()
    print('--------train_task---------')
    np.save(args.root_path+'/xdg_gate.npy',xdg_gate.detach().cpu().numpy())
    train_snn_task(net, data_set, permutation_index_list)
main()
