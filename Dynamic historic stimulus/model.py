import torch,time,os
import torch.nn as nn
import torch.nn.functional as F
cfg_fc = [512, 10] # 512
thresh = 0.2 # 0.75
lens = 0.5
decay = 0.25 # 0.75
num_classes = 10
batch_size  = 1000
# num_epochs = 100
# learning_rate = 1e-3
input_dim = 2312
time_window = 10
B_gau = 2
A = 2.5
k = A / (time_window/2)

class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < lens
        return grad_input * temp.float()

probs = 0.4
act_fun = ActFun.apply

def synapse_lin(input):
    # input[input <= time_window / 2] =  input[input <= time_window / 2] * A / (time_window/2)
    # input[input > time_window / 2] = input[input > time_window / 2] * (- A / (time_window/2)) + 2 * A + A / (time_window/2)
    return input * k

class SNN_Model(nn.Module):

    def __init__(self, num_classes=10):
        super(SNN_Model, self).__init__()

        self.ewc_lambda = 10000  # -> hyperparam: how strong to weigh EWC-loss ("regularisation strength")
        self.gamma = 1.  # -> hyperparam (online EWC): decay-term for old tasks' contribution to quadratic term
        self.online = False  # -> "online" (=single quadratic term) or "offline" (=quadratic term per task) EWC
        self.fisher_n = 5  # -> sample size for estimating FI-matrix (if "None", full pass over dataset)
        self.emp_FI = False  # -> if True, use provided labels to calculate FI ("empirical FI"); else predicted labels
        self.EWC_task_count = 0

        # self.conv1 = nn.Conv2d(2, cfg_cnn[0][1], kernel_size=3, stride=1, padding=1, )
        #
        # in_planes, out_planes, stride = cfg_cnn[1]
        # self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, )
        #
        # in_planes, out_planes, stride = cfg_cnn[2]
        # self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, )

        self.fc1 = nn.Linear(input_dim , cfg_fc[0], )
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1], )
        # self.fc3 = nn.Linear(cfg_fc[1], cfg_fc[2], )
        #self.w1 = torch.nn.Parameter(torch.randn(input_dim, cfg_fc[0]).cuda(), requires_grad=True)
        #self.w2 = torch.nn.Parameter(torch.randn(cfg_fc[0], cfg_fc[1]).cuda(), requires_grad=True)

        # self.fc3.weight.data = self.fc3.weight.data * 0.1

        self.alpha1 = torch.nn.Parameter((1e-1 * torch.ones(1)).cuda(), requires_grad=True)
        self.alpha2 = torch.nn.Parameter((1e-1 * torch.ones(1)).cuda(), requires_grad=True)

        self.eta1 = torch.nn.Parameter((1e-1 * torch.rand(1, cfg_fc[0])).cuda(), requires_grad=True)
        self.eta2 = torch.nn.Parameter((1e-1 * torch.rand(1, cfg_fc[1])).cuda(), requires_grad=True)

        self.gamma1 = torch.nn.Parameter((1e-2 * torch.rand(cfg_fc[0], cfg_fc[0])).cuda(), requires_grad=True)
        self.gamma2 = torch.nn.Parameter((1e-2 * torch.rand(cfg_fc[1], cfg_fc[1])).cuda(), requires_grad=True)

        self.beta1 = torch.nn.Parameter((1e-2 * torch.rand(1, input_dim)).cuda(), requires_grad=True)
        self.beta2 = torch.nn.Parameter((1e-2 * torch.rand(1, cfg_fc[0])).cuda(), requires_grad=True)


    def forward(self, input, gate,ID,hebb=None, win = time_window):

        # c1_mem = c1_spike = c1_sumspike = torch.zeros(batch_size, cfg_cnn[0][1], 34, 34, device=device)
        # c2_mem = c2_spike = c2_sumspike = torch.zeros(batch_size, cfg_cnn[1][1], 17, 17, device=device)
        # c3_mem = c3_spike = c3_sumspike = torch.zeros(batch_size, cfg_cnn[2][1], 8, 8, device=device)

        h1_sumstate = h1_summem = h1_den_mem = h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0]).cuda()
        h2_sumstate = h2_summem = h2_den_mem = h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1]).cuda()

        # hebb1, hebb2  = hebb

        record_h1_spike_sum=0

        for step in range(win):
            k_filter = 1

            x = input[:,step,:]
            x = x.view(batch_size, -1)

            h1_mem, h1_spike, h1_state = mem_update(self.fc1, self.alpha1, self.beta1, self.gamma1, self.eta1, x, h1_spike,
                                                 h1_mem, h1_den_mem, gate)
            h1_sumspike = h1_sumspike + h1_spike
            h1_summem = h1_summem + h1_mem
            h1_sumstate = h1_sumstate + h1_state






            record_h1_spike_sum+=h1_spike





            if(ID==True):
                h1_spike = h1_spike.mul(synapse_lin(h1_sumspike))





            h2_mem, h2_spike, h2_state = mem_update(self.fc2, self.alpha2, self.beta2, self.gamma2, self.eta2, h1_spike,
                                                 h2_spike, h2_mem, h2_den_mem)
            h2_sumspike = h2_sumspike + h2_spike
            h2_summem = h2_summem + h2_mem
            h2_sumstate = h2_sumstate + h2_state

        # outs = h2_mem / thresh
        outs = h2_sumspike / time_window
        return outs#,record_h1_spike_sum/time_window

    def estimate_fisher(self, dataset, gate,ID,permutted_paramer):
        '''After completing training on a task, estimate diagonal of Fisher Information matrix.

        [dataset]:          <DataSet> to be used to estimate FI-matrix
        [allowed_classes]:  <list> with class-indeces of 'allowed' or 'active' classes'''

        # Prepare <dict> to store estimated Fisher Information matrix
        est_fisher_info = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher_info[n] = p.detach().clone().zero_()

        # Set model to evaluation mode
        mode = self.training
        self.eval()

        # Create data-loader to give batches of size 1
        # data_loader = get_data_loader(dataset, batch_size=1, cuda=self._is_on_cuda(), collate_fn=collate_fn)
        data_loader = dataset

        # Estimate the FI-matrix for [self.fisher_n] batches of size 1
        for index, (x, y) in enumerate(data_loader):
            # break from for-loop if max number of samples has been reached
            if self.fisher_n is not None:
                if index >= self.fisher_n:
                    break
            # run forward pass of model
            # x = x.to(self._device())
            x = x[:, :, permutted_paramer].cuda()
            outputs = self(x,gate,ID)
            # output = outputs[0]
            output = outputs
            if self.emp_FI:
                # -use provided label to calculate loglikelihood --> "empirical Fisher":
                label = torch.LongTensor([y]) if type(y) == int else y
                label = label.to(self._device())
            else:
                # -use predicted label to calculate loglikelihood:
                label = output.max(1)[1]
            # calculate negative log-likelihood
            negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)

            # Calculate gradient of negative loglikelihood
            self.zero_grad()
            negloglikelihood.backward()

            # Square gradients and keep running sum
            for n, p in self.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        est_fisher_info[n] += p.grad.detach() ** 2

        # Normalize by sample size used for estimation
        est_fisher_info = {n: p / index for n, p in est_fisher_info.items()}

        # Store new values in the network
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                # -mode (=MAP parameter estimate)
                self.register_buffer('{}_EWC_prev_task{}'.format(n, "" if self.online else self.EWC_task_count + 1),
                                     p.detach().clone())
                # -precision (approximated by diagonal Fisher Information matrix)
                if self.online and self.EWC_task_count == 1:
                    existing_values = getattr(self, '{}_EWC_estimated_fisher'.format(n))
                    est_fisher_info[n] += self.gamma * existing_values
                self.register_buffer(
                    '{}_EWC_estimated_fisher{}'.format(n, "" if self.online else self.EWC_task_count + 1),
                    est_fisher_info[n])

        # If "offline EWC", increase task-count (for "online EWC", set it to 1 to indicate EWC-loss can be calculated)
        self.EWC_task_count = 1 if self.online else self.EWC_task_count + 1

        # Set model back to its initial mode
        self.train(mode=mode)

    def ewc_loss(self):
        '''Calculate EWC-loss.'''
        if self.EWC_task_count > 0:
            losses = []
            # If "offline EWC", loop over all previous tasks (if "online EWC", [EWC_task_count]=1 so only 1 iteration)
            for task in range(1, self.EWC_task_count + 1):
                for n, p in self.named_parameters():
                    if p.requires_grad:
                        # Retrieve stored mode (MAP estimate) and precision (Fisher Information matrix)
                        n = n.replace('.', '__')
                        mean = getattr(self, '{}_EWC_prev_task{}'.format(n, "" if self.online else task))
                        fisher = getattr(self, '{}_EWC_estimated_fisher{}'.format(n, "" if self.online else task))
                        # If "online EWC", apply decay-term to the running sum of the Fisher Information matrices
                        fisher = self.gamma * fisher if self.online else fisher
                        # Calculate EWC-loss
                        losses.append((fisher * (p - mean) ** 2).sum())
            # Sum EWC-loss from all parameters (and from all tasks, if "offline EWC")
            return (1. / 2) * sum(losses)
        else:
            # EWC-loss is 0 if there are no stored mode and precision yet
            return torch.tensor(0.).cuda()


def mem_update(fc, alpha, beta, gamma, eta, inputs, spike, mem, hebb,gate=1):
    state = fc(inputs)
    #state = synapse_gau(state)
    mem = (mem - spike * thresh ) * decay + state
    # mem = (1 - spike.detach()) * mem * decay + state
    now_spike = act_fun(mem - thresh)*gate
    return mem, now_spike.float(), state

