"""
Demo single-file script to train a ConvNet on CIFAR10 using SoftHebb, an unsupervised, efficient and bio-plausible
learning algorithm
"""
import os
node_name = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = node_name
import math
import warnings
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.optim.lr_scheduler import StepLR
import torchvision
import numpy as np
import random

# thre = -2

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def heaviside(x, thr=0):
    return torch.where(x>=thr, torch.ones_like(x),torch.zeros_like(x))

class SoftHebbConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            t_invert: float = 12,
    ) -> None:
        """
        Simplified implementation of Conv2d learnt with SoftHebb; an unsupervised, efficient and bio-plausible
        learning algorithm.
        This simplified implementation omits certain configurable aspects, like using a bias, groups>1, etc. which can
        be found in the full implementation in hebbconv.py
        """
        super(SoftHebbConv2d, self).__init__()
        assert groups == 1, "Simple implementation does not support groups > 1."
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = 'reflect'
        self.F_padding = (padding, padding, padding, padding)
        weight_range = 25 / math.sqrt((in_channels / groups) * kernel_size * kernel_size)
        self.weight = nn.Parameter(weight_range * torch.randn((out_channels, in_channels // groups, *self.kernel_size)))
        self.t_invert = torch.tensor(t_invert)

    def forward(self, x):
        x = F.pad(x, self.F_padding, self.padding_mode)  # pad input
        # perform conv, obtain weighted input u \in [B, OC, OH, OW]
        weighted_input = F.conv2d(x, self.weight, None, self.stride, 0, self.dilation, self.groups)

        if self.training:
            # ===== find post-synaptic activations y = sign(u)*softmax(u, dim=C), s(u)=1 - 2*I[u==max(u,dim=C)] =====
            # Post-synaptic activation, for plastic update, is weighted input passed through a softmax.
            # Non-winning neurons (those not with the highest activation) receive the negated post-synaptic activation.
            batch_size, out_channels, height_out, width_out = weighted_input.shape
            # Flatten non-competing dimensions (B, OC, OH, OW) -> (OC, B*OH*OW)
            flat_weighted_inputs = weighted_input.transpose(0, 1).reshape(out_channels, -1)
            # Compute the winner neuron for each batch element and pixel
            flat_softwta_activs = torch.softmax(self.t_invert * flat_weighted_inputs, dim=0)
            flat_softwta_activs = - flat_softwta_activs  # Turn all postsynaptic activations into anti-Hebbian
            win_neurons = torch.argmax(flat_weighted_inputs, dim=0)  # winning neuron for each pixel in each input
            competing_idx = torch.arange(flat_weighted_inputs.size(1))  # indeces of all pixel-input elements
            # Turn winner neurons' activations back to hebbian
            flat_softwta_activs[win_neurons, competing_idx] = - flat_softwta_activs[win_neurons, competing_idx]
            softwta_activs = flat_softwta_activs.view(out_channels, batch_size, height_out, width_out).transpose(0, 1)
            # ===== compute plastic update Δw = y*(x - u*w) = y*x - (y*u)*w =======================================
            # Use Convolutions to apply the plastic update. Sweep over inputs with postynaptic activations.
            # Each weighting of an input pixel & an activation pixel updates the kernel element that connected them in
            # the forward pass.
            yx = F.conv2d(
                x.transpose(0, 1),  # (B, IC, IH, IW) -> (IC, B, IH, IW)
                softwta_activs.transpose(0, 1),  # (B, OC, OH, OW) -> (OC, B, OH, OW)
                padding=0,
                stride=self.dilation,
                dilation=self.stride,
                groups=1
            ).transpose(0, 1)  # (IC, OC, KH, KW) -> (OC, IC, KH, KW)

            # sum over batch, output pixels: each kernel element will influence all batches and output pixels.
            yu = torch.sum(torch.mul(softwta_activs, weighted_input), dim=(0, 2, 3))
            delta_weight = yx - yu.view(-1, 1, 1, 1) * self.weight
            delta_weight.div_(torch.abs(delta_weight).amax() + 1e-30)  # Scale [min/max , 1]
            self.weight.grad = delta_weight  # store in grad to be used with common optimizers

        return weighted_input


class DeepSoftHebb(nn.Module):
    def __init__(self):
        super(DeepSoftHebb, self).__init__()
        # block 1
        self.bn1 = nn.BatchNorm2d(3, affine=False)
        self.conv1 = SoftHebbConv2d(in_channels=3, out_channels=96, kernel_size=5, padding=2, t_invert=1,)
        self.activ1 = Triangle(power=0.7)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        # block 2
        self.bn2 = nn.BatchNorm2d(96, affine=False)
        self.conv2 = SoftHebbConv2d(in_channels=96, out_channels=384, kernel_size=3, padding=1, t_invert=0.65,)
        self.activ2 = Triangle(power=1.4)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        # block 3
        self.bn3 = nn.BatchNorm2d(384, affine=False)
        self.conv3 = SoftHebbConv2d(in_channels=384, out_channels=1536, kernel_size=3, padding=1, t_invert=0.25,)
        self.activ3 = Triangle(power=1.)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        # block 4
        self.flatten = nn.Flatten()
        self.projection_dim = 10000
        # Method 1
        # self.sparse_projection = (torch.rand(24576,self.projection_dim) < 0.001).float()
        # Method 2
        self.sparse_projection = nn.Parameter(torch.rand(24576,self.projection_dim).cuda())
        self.sparse_mask0 = torch.rand(5, self.projection_dim).float().cuda()
        self.sparse_mask = (self.sparse_mask0 < 0.1).float().cuda()
        self.classifier = nn.Linear(self.projection_dim, 10, bias=False)
        self.classifier.weight.data = 0.11048543456039805 * torch.rand(10, self.projection_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, is_training=False, task_id = None):
        # block 1
        out = self.pool1(self.activ1(self.conv1(self.bn1(x))))
        # block 2
        out = self.pool2(self.activ2(self.conv2(self.bn2(out))))
        # block 3
        out = self.pool3(self.activ3(self.conv3(self.bn3(out)))).detach_()
        # block 4
        if not is_training:
            out_flatten = self.flatten(out)
            acti_neuron = ((self.sparse_mask0[task_id] - 0.5) * self.sparse_mask[task_id]).view(1,self.projection_dim).to(device)
            # print('***', acti_neuron.shape)
            # print(acti_neuron)
            S = (torch.mean(out_flatten,dim=0) @ self.sparse_projection)#.detach()
            # print('***s max', S.max())
            # print('***s min', S.min())   1  fixed thre  2  top k
            acti_neuron_fire = heaviside(S,thr=0.7*torch.mean(S)) * acti_neuron #.detach()
            #print('S',heaviside(S))
            delta_w = torch.mean(out_flatten,dim=0).view(24576,1) @ acti_neuron_fire * 0.1 #
            projection = out_flatten @ (self.sparse_projection + delta_w).to(device)
            projection_winner = projection * self.sparse_mask[task_id]
            return self.classifier(self.dropout(projection_winner))
        else:
            return None

class Triangle(nn.Module):
    def __init__(self, power: float = 1, inplace: bool = True):
        super(Triangle, self).__init__()
        self.inplace = inplace
        self.power = power

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input - torch.mean(input.data, axis=1, keepdims=True)
        return F.relu(input, inplace=self.inplace) ** self.power


class WeightNormDependentLR(optim.lr_scheduler._LRScheduler):
    """
    Custom Learning Rate Scheduler for unsupervised training of SoftHebb Convolutional blocks.
    Difference between current neuron norm and theoretical converged norm (=1) scales the initial lr.
    """

    def __init__(self, optimizer, power_lr, last_epoch=-1, verbose=False):
        self.optimizer = optimizer
        self.initial_lr_groups = [group['lr'] for group in self.optimizer.param_groups]  # store initial lrs
        self.power_lr = power_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        new_lr = []
        for i, group in enumerate(self.optimizer.param_groups):
            for param in group['params']:
                # difference between current neuron norm and theoretical converged norm (=1) scales the initial lr
                # initial_lr * |neuron_norm - 1| ** 0.5
                norm_diff = torch.abs(torch.linalg.norm(param.view(param.shape[0], -1), dim=1, ord=2) - 1) + 1e-10
                new_lr.append(self.initial_lr_groups[i] * (norm_diff ** self.power_lr)[:, None, None, None])
        return new_lr


class TensorLRSGD(optim.SGD):
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step, using a non-scalar (tensor) learning rate.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                p.add_(-group['lr'] * d_p)
        return loss

class CustomStepLR(StepLR):
    """
    Custom Learning Rate schedule with step functions for supervised training of linear readout (classifier)
    """

    def __init__(self, optimizer, nb_epochs):
        threshold_ratios = [0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.step_thresold = [int(nb_epochs * r) for r in threshold_ratios]
        super().__init__(optimizer, -1, False)

    def get_lr(self):
        if self.last_epoch in self.step_thresold:
            return [group['lr'] * 0.5
                    for group in self.optimizer.param_groups]
        return [group['lr'] for group in self.optimizer.param_groups]


class FastCIFAR10(torchvision.datasets.CIFAR10):
    """
    Improves performance of training on CIFAR10 by removing the PIL interface and pre-loading on the GPU (2-3x speedup).

    Taken from https://github.com/y0ast/pytorch-snippets/tree/main/fast_mnist
    """

    def __init__(self, *args, **kwargs):
        device = kwargs.pop('device', "cpu")
        super().__init__(*args, **kwargs)

        self.data = torch.tensor(self.data, dtype=torch.float, device=device).div_(255)
        self.data = torch.movedim(self.data, -1, 1)  # -> set dim to: (batch, channels, height, width)
        self.targets = torch.tensor(self.targets, device=device)

    def __getitem__(self, index: int):
        """
        Parameters
        ----------
        index : int
            Index of the element to be returned

        Returns
        -------
            tuple: (image, target) where target is the index of the target class
        """
        img = self.data[index]
        target = self.targets[index]

        return img, target


# Main training loop CIFAR10
if __name__ == "__main__":
    device = torch.device('cuda:0')
    setup_seed(98) #98
    model = DeepSoftHebb()
    model.to(device)

    unsup_optimizer = TensorLRSGD([
        {"params": model.conv1.parameters(), "lr": -0.08, },  # SGD does descent, so set lr to negative
        {"params": model.conv2.parameters(), "lr": -0.005, },
        {"params": model.conv3.parameters(), "lr": -0.01, },
    ], lr=0)
    unsup_lr_scheduler = WeightNormDependentLR(unsup_optimizer, power_lr=0.5)
    
    parameter_list = [model.classifier.weight,  model.sparse_projection]
    sup_optimizer = optim.Adam(parameter_list, lr=0.001)
    sup_lr_scheduler = CustomStepLR(sup_optimizer, nb_epochs=50)
    criterion = nn.CrossEntropyLoss()

    trainset = FastCIFAR10('/data/xmk2/Postdoctor/Hybrid_BTSP_softhebb-main/cifar10/cifar-10-batches-py', train=True, download=True)
    unsup_trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, )
    sup_trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, )

    testset = FastCIFAR10('/data/xmk2/Postdoctor/Hybrid_BTSP_softhebb-main/cifar10/cifar-10-batches-py', train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

    # Function to get indices for class labels
    def get_indices(dataset, class_labels):
        indices = []
        for i, (_, label) in enumerate(dataset):
            if label in class_labels:
                indices.append(i)
        return indices


    # Create subsets and their corresponding DataLoaders
    subsets_trainloaders = []
    subsets_testloaders = []
    from torch.utils.data import DataLoader, Subset, random_split
    for i in range(0, 10, 2):
        train_indices = get_indices(trainset, list(range(i, i + 2)))
        test_indices = get_indices(testset, list(range(i, i + 2)))
    
        train_subset = Subset(trainset, train_indices)
        test_subset = Subset(testset, test_indices)
    
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=1000, shuffle=False)
    
        subsets_trainloaders.append(train_loader)
        subsets_testloaders.append(test_loader)

    # Unsupervised training with SoftHebb
    running_loss = 0.0
    num_epoch = 20
    for sub_train_loader in subsets_trainloaders:
        for i, data in enumerate(sub_train_loader, 0):
            inputs, _ = data
            inputs = inputs.to(device)
    
            # zero the parameter gradients
            unsup_optimizer.zero_grad()
    
            # forward + update computation
            with torch.no_grad():
                outputs = model(inputs, is_training=True)
    
            # optimize
            unsup_optimizer.step()
            unsup_lr_scheduler.step()
    print('Training for the softhebb is finished!')
    # Supervised training of classifier
    # set requires grad false and eval mode for all modules but classifier
    unsup_optimizer.zero_grad()
    model.conv1.requires_grad = False
    model.conv2.requires_grad = False
    model.conv3.requires_grad = False
    model.conv1.eval()
    model.conv2.eval()
    model.conv3.eval()
    model.bn1.eval()
    model.bn2.eval()
    model.bn3.eval()
    
    for task_id, sub_train_loader in enumerate(subsets_trainloaders):
        
        for epoch in range(num_epoch):
            model.classifier.train()
            model.dropout.train()
            running_loss = 0.0
            correct = 0
            total = 0
            start_time = time.time()
            for i, data in enumerate(sub_train_loader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
    
                # zero the parameter gradients
                sup_optimizer.zero_grad()
                
                # forward + backward + optimize
                outputs = model(inputs,task_id=task_id)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                loss.backward()
                sup_optimizer.step()
    
                # compute training statistics
                running_loss += loss.item()
                if epoch % 2 == 0 or epoch == num_epoch-1:
                    total += labels.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
            
            
        # Evaluation on test set
        if epoch % 5 == 0 or epoch == num_epoch-1:
            print('time eslapses',time.time()-start_time)
            start_time = time.time()
            print(f'Accuracy of the network on the train images: {100 * correct // total} %')
            print(f'[{epoch + 1}] loss: {running_loss / total:.3f}')

            # on the test set
            model.eval()
            running_loss = 0.
            correct = 0
            total = 0
            # since we're not training, we don't need to calculate the gradients for our outputs
            sub_testloader = subsets_testloaders[task_id]
            with torch.no_grad():
                for data in sub_testloader:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    # calculate outputs by running images through the network
                    outputs = model(images,task_id=task_id)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()

            print(f'Subtask: accuracy of the network on the Subtask test images: {100 * correct / total} %')
            print(f'Subtask: test loss: {running_loss / total:.3f}')

        model.eval()
        running_loss = 0.
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for task_id, sub_daloader in enumerate(subsets_testloaders):
                correct_sub, total_sub = 0, 0.
                for data in sub_daloader:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    # calculate outputs by running images through the network
                    outputs = model(images,task_id=task_id)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total_sub += labels.size(0)
                    correct_sub += (predicted == labels).sum().item()
                    
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()
                correct += correct_sub
                total += total_sub
                print('Subtask', task_id, 'accuracy', 100*correct_sub/total_sub)
        print(f'Finish training for a subtask! Accuracy of the network on the 10000 test images: {100 * correct / total} %')
        print(f'test loss: {running_loss / total:.3f}')
        sup_lr_scheduler.step()