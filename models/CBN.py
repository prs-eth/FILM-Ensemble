import torch
import torch.nn as nn
from torch.autograd import Variable


class CBN2D(nn.Module):

    def __init__(self, n_ensemble, num_features, eps=1e-5, momentum=0.9, cbn_is_training=True, name='cbn',
                 trainable=True, cbn_gain=1.0, init_type='xavier'):
        super(CBN2D, self).__init__()
        self.n_ensemble = n_ensemble
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.cbn_is_training = cbn_is_training
        self.name = name
        self.trainable = trainable
        self.cbn_gain = cbn_gain
        self.init_type = init_type

        # Affine transform parameters
        self.gamma = nn.Parameter(torch.ones(n_ensemble, num_features), requires_grad=self.trainable)
        self.beta = nn.Parameter(torch.zeros(n_ensemble, num_features), requires_grad=self.trainable)

        # Running mean and variance, these parameters are not trained by backprop
        self.running_mean = nn.Parameter(torch.Tensor(n_ensemble, num_features), requires_grad=False)
        self.running_var = nn.Parameter(torch.Tensor(n_ensemble, num_features), requires_grad=False)
        self.num_batches_tracked = nn.Parameter(torch.tensor(0), requires_grad=False)

        # Parameter initilization
        self.reset_parameters()

        # Initialize weights using Xavier initialization and biases with constant value
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0.1)

        # Initialize gamma and beta
        if self.init_type == 'bernoulli':
            with torch.no_grad():
                gamma_init = torch.ones(n_ensemble, num_features)
                beta_int = torch.ones(n_ensemble, num_features)
                gamma_init.bernoulli_(0.5).mul_(2).add_(-1).mul_(self.cbn_gain)
                beta_int.bernoulli_(0.5).mul_(2).add_(-1).mul_(self.cbn_gain)
                self.gamma = nn.Parameter(gamma_init, requires_grad=self.trainable)
                self.beta = nn.Parameter(beta_int, requires_grad=self.trainable)
        elif self.init_type == 'xavier':
            nn.init.xavier_uniform_(self.gamma, gain=self.cbn_gain)
            nn.init.xavier_uniform_(self.beta, gain=self.cbn_gain)
        else:
            print('WARNING: Wrong init type - CBNs are not initilized!!!')

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.fill_(0)

    def reset_parameters(self):
        self.reset_running_stats()

    def set_cbn_mode(self, cbn_is_training):
        self.cbn_is_training = cbn_is_training
        return cbn_is_training

    def batch_norm(self, input, running_mean, running_var, gammas, betas, is_training, exponential_average_factor, eps):
        N, C, H, W = input.size()
        B = N // self.n_ensemble

        # mean and variance per member and channel, shape (M, C)
        mean = input.view(B, self.n_ensemble, C, W, H).mean(dim=(0, 3, 4))
        variance = input.view(B, self.n_ensemble, C, W, H).var(dim=(0, 3, 4))

        if is_training:
            # Compute running mean and variance per member and channel, shape (M, C)
            running_mean = running_mean * (1 - exponential_average_factor) + mean * exponential_average_factor
            running_var = running_var * (1 - exponential_average_factor) + variance * exponential_average_factor

            # Training mode, normalize the data using its mean and variance
            X_hat = (input - mean.tile(B, 1)[:, :, None, None]) * 1.0 / torch.sqrt(
                variance.tile(B, 1)[:, :, None, None] + eps)
        else:
            # Test mode, normalize the data using the running mean and variance
            X_hat = (input - running_mean.tile(B, 1)[:, :, None, None]) * 1.0 / torch.sqrt(
                running_var.tile(B, 1)[:, :, None, None] + eps)

        # Scale and shift
        out = gammas.tile(B, 1)[:, :, None, None] * X_hat + betas.tile(B, 1)[:, :, None, None]

        return out, running_mean, running_var

    def forward(self, input):
        exponential_average_factor = 0.0
        if self.cbn_is_training:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked
            else:  # use exponential moving average
                exponential_average_factor = 1 - self.momentum

        # Standard batch normalization
        out, running_mean, running_var = self.batch_norm(
            input, self.running_mean, self.running_var, self.gamma, self.beta, self.cbn_is_training,
            exponential_average_factor, self.eps)

        if self.cbn_is_training:
            self.running_mean[:] = running_mean[:]
            self.running_var[:] = running_var[:]

        return out


class CBN1D(nn.Module):

    def __init__(self, n_ensemble, num_features, eps=1e-5, momentum=0.9, cbn_is_training=True, name='cbn',
                 trainable=True, cbn_gain=1.0, init_type='xaiver'):
        super(CBN1D, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.cbn_is_training = cbn_is_training
        self.name = name
        self.trainable = trainable
        self.cbn_gain = cbn_gain
        self.init_type = init_type

        self.active_task = 0

        # Affine transform parameters
        self.gamma = nn.Parameter(torch.ones(n_ensemble, num_features), requires_grad=self.trainable)
        self.beta = nn.Parameter(torch.zeros(n_ensemble, num_features), requires_grad=self.trainable)

        # Running mean and variance, these parameters are not trained by backprop
        self.running_mean = nn.Parameter(torch.Tensor(n_ensemble, num_features), requires_grad=False)
        self.running_var = nn.Parameter(torch.Tensor(n_ensemble, num_features), requires_grad=False)
        self.num_batches_tracked = nn.Parameter(torch.Tensor(n_ensemble, 1), requires_grad=False)

        # Parameter initilization
        self.reset_parameters()

        # Initialize weights using Xavier initialization and biases with constant value
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0.1)

        # Initialize gamma and beta
        if self.init_type == 'bernoulli':
            with torch.no_grad():
                gamma_init = torch.ones(n_ensemble, num_features)
                beta_int = torch.ones(n_ensemble, num_features)
                gamma_init.bernoulli_(0.5).mul_(2).add_(-1).mul_(self.cbn_gain)
                beta_int.bernoulli_(0.5).mul_(2).add_(-1).mul_(self.cbn_gain)
                self.gamma = nn.Parameter(gamma_init, requires_grad=self.trainable)
                self.beta = nn.Parameter(beta_int, requires_grad=self.trainable)
        elif self.init_type == 'xavier':
            nn.init.xavier_uniform_(self.gamma, gain=self.cbn_gain)
            nn.init.xavier_uniform_(self.beta, gain=self.cbn_gain)
        else:
            print('WARNING: Wrong init type - CBNs are not initilized!!!')


    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()

    def set_active_task(self, active_task):
        self.active_task = active_task
        return active_task

    def set_cbn_mode(self, cbn_is_training):
        self.cbn_is_training = cbn_is_training
        return cbn_is_training

    def batch_norm(self, input, running_mean, running_var, gammas, betas,
                   is_training, exponential_average_factor, eps):
        # Extract the dimensions
        N, C = input.size()

        # Mini-batch mean and variance
        mean = input.mean(dim=0)
        variance = input.var(dim=0)

        # Normalize
        if is_training:
            # Compute running mean and variance
            running_mean = running_mean * (1 - exponential_average_factor) + mean * exponential_average_factor
            running_var = running_var * (1 - exponential_average_factor) + variance * exponential_average_factor

            # Training mode, normalize the data using its mean and variance
            X_hat = (input - mean.view(1, C).expand((N, C))) * 1.0 / torch.sqrt(
                variance.view(1, C).expand((N, C)) + eps)
        else:
            # Test mode, normalize the data using the running mean and variance
            X_hat = (input - running_mean.view(1, C).expand((N, C))) * 1.0 / torch.sqrt(
                running_var.view(1, C).expand((N, C)) + eps)

        # Scale and shift
        out = gammas.contiguous().view(N, C).expand((N, C)) * X_hat + betas.contiguous().view(N, C).expand((N, C))

        return out, running_mean, running_var

    def forward(self, input):

        N, C = input.size()

        exponential_average_factor = 0.0
        if self.cbn_is_training:
            self.num_batches_tracked[self.active_task] += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked[self.active_task]
            else:  # use exponential moving average
                exponential_average_factor = 1 - self.momentum

        # Choose gamma and beta depends on active_task
        gamma_cloned = self.gamma[self.active_task, :].view(1, C).expand(N, C).clone()
        beta_cloned = self.beta[self.active_task, :].view(1, C).expand(N, C).clone()

        # Standard batch normalization
        out, running_mean, running_var = self.batch_norm(input, self.running_mean[self.active_task, :],
                                                         self.running_var[self.active_task, :],
                                                         gamma_cloned, beta_cloned,
                                                         self.cbn_is_training, exponential_average_factor, self.eps)

        if self.cbn_is_training:
            self.running_mean.data[self.active_task, :] = running_mean.data
            self.running_var.data[self.active_task, :] = running_var.data

        return out


if __name__ == '__main__':
    model = CBN1D(5, 128).cuda()
    x = torch.ones([4, 128])
    x = Variable(x.cuda())

    model.eval()
    print(model(x))

    # print(model.state_dict)
    # print(model.cbn_is_training)