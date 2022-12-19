'''Train FiLM-Ensmeble for CIFAR10/Cifar100/ with PyTorch.'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from loguru import logger
import os
import argparse
import random
import time
from models.vgg_film import vgg_cbn
from models.resnet_film import ResNet18_FILM, ResNet34_FILM

from utils_uncertainty import _ECELoss, function_space_analysis

def seed_everything(seed: int):
    """From https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/utilities/seed.html#seed_everything"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print('seeding everything w/ seed', seed)

parser = argparse.ArgumentParser(description='FilM-Ensmeble CIFAR10/CIFAR100 Training')
parser.add_argument('--dataset', default='Cifar10', type=str, help='Cifar10/Cifar100')
parser.add_argument('--datadir', default='./data', type=str, help='dataset directory')
parser.add_argument('--ensemble', '-e', default=2, type=int, help='number of ensemble members')
parser.add_argument('--cbn', '-c', default='v1', type=str, help='CBN version: v2, v3')
parser.add_argument('--batch-mode', '-b', default='default', type=str, help='batching version: divide, default')
parser.add_argument('--is-cbn-trainable', default=True, type=bool, help='is_film gammas/betas are trainable trainable')
parser.add_argument('--use-consensus-pooling', default=False, type=bool, help='consensus pooling or mean for the final prediction')
parser.add_argument('--net-type', default='vit', type=str, help='net type: VGG11, VGG13, VGG16, VGG19, Resnet18, Resnet34')
parser.add_argument('--cln_gain', default=5.0, type=float, help='film initilization gain factor')
parser.add_argument('--cbn_gain', default=1.0, type=float, help='film initilization gain factor')
parser.add_argument('--init-type', default='xavier', type=str, help='film initilization type: xavier, bernoulli')
parser.add_argument('--drop-rate', default=0.0, type=float, help='Dropout rate')
parser.add_argument('--max-epoch', default=400, type=int, help='number of epochs to train')
parser.add_argument('--optim-config', default='C', type=str, help='optimizer configuration: A, B, C')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--grad-clip', default=5, type=float, help='max gradient value allowed')
parser.add_argument('--eval', action='store_true', help='mode: eval or train')
parser.add_argument('--measure-time', action='store_true', help='measure inference time')
parser.add_argument('--resume', '-r', default=None, type=str, help='resume from checkpoint')
parser.add_argument('--save-model', default=True, type=bool, help='where the trained model is saved')
parser.add_argument('--save-dir', '-s', type=str, help='resume from checkpoint', default='./checkpoint/')
parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--wandb', default=False, action='store_true', help='use wandb')
parser.add_argument('--aug', default=True, action='store_true', help='use randomaug')
parser.add_argument('--num_heads', default = 8, type=int, help = 'number of heads in Transformer')

args = parser.parse_args()
print(args)
seed_everything(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
num_ensembles = args.ensemble

watermark = "AffineFIX2_head{}_{}_lr{}_ens{}_gain{}_clip{}".format(args.num_heads, args.net_type, args.lr, args.ensemble, args.cln_gain, args.grad_clip)
if args.wandb:
    import wandb
    wandb.init(project='film', dir=args.save_dir, name=watermark)
    wandb.config.update(vars(args))


#-----------------------Data transforms -------------------------------------
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

# Add RandAugment with N, M(hyperparameter)
if args.aug:  
    from randomaug import RandAugment
    N = 2; M = 14;
    transform_train.transforms.insert(0, RandAugment(N, M))

#-------------------------------Datasets----------------------------
if args.dataset=='Cifar10':
    trainset = torchvision.datasets.CIFAR10(root=args.datadir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=args.datadir, train=False, download=True, transform=transform_test)
    num_classes = 10

elif args.dataset=='Cifar100':
    trainset = torchvision.datasets.CIFAR100(root=args.datadir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=args.datadir, train=False, download=True, transform=transform_test)
    num_classes = 100

#----------------------------Dataloaders---------------------------
if args.batch_mode == 'divide':
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128//num_ensembles, shuffle=True, num_workers=8)
else:
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)

testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=8)



#---------------------------------Model---------------------------
print('==> Building model..')

if args.net_type in ['VGG11', 'VGG13', 'VGG16', 'VGG19']:
    cfg = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                  'M'],
    }
    net = vgg_cbn(cfg=cfg[args.net_type], pretrained=False, task_count=num_ensembles,
                  is_cbn_trainable=args.is_cbn_trainable, cbn_gain=args.cbn_gain,
                  drop_rate=args.drop_rate,
                  cbn_version=args.cbn, num_classes=num_classes, init_type=args.init_type)
elif args.net_type == 'Resnet18':
    net = ResNet18_FILM(task_count=num_ensembles, cbn_gain=args.cbn_gain, is_cbn_trainable=args.is_cbn_trainable,
                        num_classes=num_classes)
elif args.net_type == 'Resnet34':
    net = ResNet34_FILM(task_count=num_ensembles, cbn_gain=args.cbn_gain, is_cbn_trainable=args.is_cbn_trainable,
                        num_classes=num_classes
                        )
if device == 'cuda':
     net = torch.nn.DataParallel(net)
     cudnn.benchmark = True

net.to(device)
print(net)

model_parameters = filter(lambda p: p.requires_grad, net.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('Number of model parameters: ', params)
if args.wandb:
    wandb.config.update({"number_params": params})
if args.eval and args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.resume), 'Error: no checkpoint file found!'
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# loss
criterion = nn.CrossEntropyLoss()
nll_loss = nn.NLLLoss()

# Optimizer and lr scheduler
if args.optim_config == 'A':
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)
elif args.optim_config == 'B':
    # This is optimizer and scheduler configuration in WideResnet paper
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.2)
elif args.optim_config == 'C':
    optimizer = optim.Adam(net.parameters(), lr=args.lr,
                          betas=(0.9, 0.999), weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)

print(optimizer)
print(scheduler)

# Ensemble mode changer
def change_cbn_mode(m):
    if hasattr(m, 'cbn_is_training'):
        m.set_cbn_mode(cbn_training_mode)

# Uncertanity criteria
if device == 'cuda':
    ece_criterion = _ECELoss().cuda()
else:
    ece_criterion = _ECELoss()

# Subnetwork independency analysis
indAnaysisF = function_space_analysis()


# Training
def train(net, epoch):
    print('\nEpoch: %d' % epoch)
    net.train()

    # Set the CBN mode
    global cbn_training_mode
    cbn_training_mode = True
    net = net.apply(change_cbn_mode)

    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        B = inputs.shape[0]
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        outputs = outputs.view(B * num_ensembles, -1)  # (B, M, 10) -> (B * M, 10)

        # repeat targets so that ensemble members are trained concurrently
        targets = targets.repeat_interleave(num_ensembles)
        loss = criterion(outputs, targets)
        loss.backward()

        # Apply gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)

        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #             % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    if args.wandb:
        wandb.log({'loss/train': train_loss / (batch_idx + 1), 'acc/train': 100. * correct / total}, epoch)

    return net


# Testing
def test(net, loader, epoch, num_used_members, measure_batch_time=False):
    print(str('Number of used members for perdiction: ' + str(num_used_members) ) )

    global best_acc
    net.eval()

    # Set the CBN mode
    global cbn_training_mode
    cbn_training_mode = False
    net = net.apply(change_cbn_mode)

    test_loss = 0
    correct = 0
    total = 0
    outputs_all = list()
    outputs_all_mean = list()
    targets_all = list()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)

            if measure_batch_time:
                start_time = time.time()

            outputs = net(inputs).view(inputs.shape[0], num_used_members, -1)

            # average ensemble members softmax output
            output_probs_mean = torch.softmax(outputs, 2).mean(1)  # shape [B, 10]

            if measure_batch_time:
                end_time = time.time()
                print('Inference time: {:.4f}'.format(end_time - start_time))


            loss = nll_loss(torch.log(output_probs_mean), targets)

            outputs_all.append(outputs)
            outputs_all_mean.append(outputs.mean(1))
            targets_all.append(targets)

            test_loss += loss.item()
            _, predicted = output_probs_mean.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        outputs_all_mean = torch.cat(outputs_all_mean)
        outputs_all = torch.cat(outputs_all)
        targets_all = torch.cat(targets_all)

        # Calibration performances
        ece, accs, confs = ece_criterion(outputs_all_mean, targets_all)

        print('Epoch:%.1f Val: Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (epoch, test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        print('ECE: {:.4f}'.format(ece.item()))

        metrics = {
        'loss/test': test_loss / (batch_idx + 1),
        'acc/test': 100. * correct / total,
        'ECE/test': ece.item(),
        'best_acc/test': best_acc
        }

        if args.save_model:
            # Save checkpoint.
            acc = 100. * correct / total
            if acc > best_acc:
                logger.info('Saving the model here: ', args.save_dir)
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }
                if not os.path.isdir(args.save_dir):
                    os.mkdir(args.save_dir)
                torch.save(state, os.path.join(args.save_dir, 'ckpt_bestVal_Film-Enformer_seed{}_'.format(args.seed)+watermark+'.pth'))
                best_acc = acc
            elif epoch == args.max_epoch-1:
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }
                torch.save(state,
                            os.path.join(args.save_dir, 'ckpt_Film-Enformer_seed{}'.format(args.seed)+watermark+'.pth'))
                                        

    if args.wandb:
        wandb.log(metrics, epoch)

    return outputs_all_mean, targets_all


# -------------MAIN---------------
if  args.measure_time:
    test(net=net, loader=testloader, epoch=0, num_used_members=args.ensemble, measure_batch_time=True)
elif args.eval:
    test(net=net, loader=testloader, epoch=0, num_used_members=args.ensemble)
else:
    for epoch in range(start_epoch, start_epoch + args.max_epoch):
        net = train(net, epoch)
        scheduler.step()
        test(net, testloader, epoch, num_used_members=args.ensemble)