'''
    main process for a Lottery Tickets experiments
'''
import argparse
import collections
import os
import pdb
import pickle
import random
import shutil
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from pruner import *
from utils import *
# from advertorch.utils import NormalizeByChannelMeanStd

parser = argparse.ArgumentParser(
    description='PyTorch Lottery Tickets Experiments')

##################################### Dataset #################################################
parser.add_argument('--data',
                    type=str,
                    default='../data',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--input_size',
                    type=int,
                    default=32,
                    help='size of input images')

##################################### Architecture ############################################
parser.add_argument('--arch',
                    type=str,
                    default='resnet20s',
                    help='model architecture')
parser.add_argument('--imagenet_arch',
                    action="store_true",
                    help="architecture for imagenet size samples")

##################################### General setting ############################################
parser.add_argument('--seed', default=None, type=int, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--workers',
                    type=int,
                    default=4,
                    help='number of workers in dataloader')
parser.add_argument('--resume',
                    action="store_true",
                    help="resume from checkpoint")
parser.add_argument('--checkpoint',
                    type=str,
                    default=None,
                    help='checkpoint file')
parser.add_argument('--save_dir',
                    help='The directory used to save the trained models',
                    default=None,
                    type=str)

##################################### Training setting #################################################
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--lr',
                    default=0.1,
                    type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay',
                    default=1e-4,
                    type=float,
                    help='weight decay')
parser.add_argument('--epochs',
                    default=250,
                    type=int,
                    help='number of total epochs to run')
parser.add_argument('--warmup', default=0, type=int, help='warm up epochs')
parser.add_argument('--print_freq',
                    default=50,
                    type=int,
                    help='print frequency')
parser.add_argument('--decreasing_lr',
                    default='91,136',
                    help='decreasing strategy')

##################################### Pruning setting #################################################
parser.add_argument('--pruning_times',
                    default=20,
                    type=int,
                    help='overall times of pruning')
parser.add_argument('--rate', default=0.2, type=float, help='pruning rate')
parser.add_argument('--prune_type',
                    default='rewind_lt',
                    type=str,
                    help='IMP type (lt, pt or rewind_lt)')
parser.add_argument('--random_prune',
                    action='store_true',
                    help='whether using random prune')
parser.add_argument('--rewind_epoch',
                    default=3,
                    type=int,
                    help='rewind checkpoint')
parser.add_argument("--density", default=0.01, type=float)

best_sa = 0


def main():
    global args, best_sa
    args = parser.parse_args()
    print(args)

    torch.cuda.set_device(int(args.gpu))

    if args.seed:
        setup_seed(args.seed)

    # prepare dataset
    model, train_loader, val_loader, test_loader = setup_model_dataset(args)
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

    if args.prune_type == 'lt':
        print('lottery tickets setting (rewind to the same random init)')
        initalization = deepcopy(model.state_dict())
    elif args.prune_type == 'pt':
        print('lottery tickets from best dense weight')
        initalization = None
    elif args.prune_type == 'rewind_lt':
        print('lottery tickets with early weight rewinding')
        initalization = None
    else:
        raise ValueError('unknown prune_type')

    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(args.epochs / 2),
                    int(args.epochs * 3 / 4)],
        gamma=0.1)

    if args.resume:
        print('resume from checkpoint {}'.format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint,
                                map_location=torch.device('cuda:' +
                                                          str(args.gpu)))
        best_sa = checkpoint['best_sa']
        start_epoch = checkpoint['epoch']
        all_result = checkpoint['result']
        start_state = checkpoint['state']

        if start_state > 0:
            current_mask = extract_mask(checkpoint['state_dict'])
            prune_model_custom(model, current_mask)
            check_sparsity(model)
            optimizer = torch.optim.SGD(model.parameters(),
                                        args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=decreasing_lr, gamma=0.1)

        model.load_state_dict(checkpoint['state_dict'])
        # adding an extra forward process to enable the masks
        x_rand = torch.rand(1, 3, args.input_size, args.input_size).cuda()
        model.eval()
        with torch.no_grad:
            model(x_rand)

        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        initalization = checkpoint['init_weight']
        print('loading state:', start_state)
        print('loading from epoch: ', start_epoch, 'best_sa=', best_sa)

    else:
        all_result = {}
        all_result['train_ta'] = []
        all_result['test_ta'] = []
        all_result['val_ta'] = []

        start_epoch = 0
        start_state = 0

    print(
        '######################################## Start Standard Training Iterative Pruning ########################################'
    )

    for state in range(start_state, args.pruning_times):

        print('******************************************')
        print('pruning state', state)
        print('******************************************')
        # create saving folder for each state
        density = check_sparsity(model) / 100.0
        timestr = time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
        save_dir = os.path.join(args.save_dir, f"density_{density:.2f}",
                                args.dataset, args.arch, 'lth', str(args.seed),
                                timestr)
        os.makedirs(save_dir, exist_ok=True)
        # reset best acc for each state
        df = collections.defaultdict(list)
        best_acc = 0.0
        for epoch in range(start_epoch, args.epochs):

            print(optimizer.state_dict()['param_groups'][0]['lr'])
            acc, loss = train(train_loader, model, criterion, optimizer, epoch)

            if state == 0:
                if (epoch + 1
                    ) == args.rewind_epoch and args.prune_type == 'rewind_lt':
                    # torch.save(
                    # model.state_dict(),
                    # os.path.join(
                    # args.save_dir,
                    # 'epoch_{}_rewind_weight.pt'.format(epoch + 1)))
                    # if args.prune_type == 'rewind_lt':
                    initalization = deepcopy(model.state_dict())
                    torch.save(initalization,
                               os.path.join(save_dir, 'init.pth.tar'))

            # evaluate on validation set
            tacc = validate(val_loader, model, criterion)
            # evaluate on test set
            test_tacc = validate(test_loader, model, criterion)

            scheduler.step()

            df['loss'].append(loss)
            df['val_acc'].append(tacc)
            df['test_acc'].append(test_tacc)
            df['epoch'].append(epoch)
            # all_result['train_ta'].append(acc)
            # all_result['val_ta'].append(tacc)
            # all_result['test_ta'].append(test_tacc)

            # remember best prec@1 and save checkpoint
            # is_best_sa = tacc > best_sa
            # best_sa = max(tacc, best_sa)
            if tacc > best_acc:
                best_acc = tacc
                torch.save(
                    {
                        "epoch": epoch + 1,
                        'state_dict': model.state_dict()
                    }, os.path.join(save_dir, 'model.pth'))

            # save_checkpoint(
            # {
            # 'state': state,
            # 'result': all_result,
            # 'epoch': epoch + 1,
            # 'state_dict': model.state_dict(),
            # 'best_sa': best_sa,
            # 'optimizer': optimizer.state_dict(),
            # 'scheduler': scheduler.state_dict(),
            # 'init_weight': initalization
            # },
            # is_SA_best=is_best_sa,
            # pruning=state,
            # save_path=args.save_dir)

            # # plot training curve
            # plt.plot(all_result['train_ta'], label='train_acc')
            # plt.plot(all_result['val_ta'], label='val_acc')
            # plt.plot(all_result['test_ta'], label='test_acc')
            # plt.legend()
            # plt.savefig(
            # os.path.join(args.save_dir,
            # str(state) + 'net_train.png'))
            # plt.close()

        #report resultj
        check_sparsity(model)
        # val_pick_best_epoch = np.argmax(np.array(all_result['val_ta']))
        # print('* best SA = {}, Epoch = {}'.format(
        # all_result['test_ta'][val_pick_best_epoch],
        # val_pick_best_epoch + 1))
        # save csv
        df = pd.DataFrame.from_dict(df)
        df.to_csv(os.path.join(save_dir, 'results.csv'))

        # all_result = {}
        # all_result['train_ta'] = []
        # all_result['test_ta'] = []
        # all_result['val_ta'] = []
        best_sa = 0
        start_epoch = 0

        # if args.prune_type == 'pt':
        # print('* loading pretrained weight')
        # initalization = torch.load(
        # os.path.join(args.save_dir, '0model_SA_best.pth.tar'),
        # map_location=torch.device('cuda:' +
        # str(args.gpu)))['state_dict']

        # #pruning and rewind
        # if args.random_prune:
        # print('random pruning')
        # pruning_model_random(model, args.rate)
        # else:
        print('L1 pruning')
        pruning_model(model, args.rate)

        remain_weight = check_sparsity(model)
        current_mask = extract_mask(model.state_dict())
        remove_prune(model)

        # weight rewinding
        model.load_state_dict(initalization)
        prune_model_custom(model, current_mask)
        optimizer = torch.optim.SGD(model.parameters(),
                                    args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=decreasing_lr, gamma=0.1)

        if args.prune_type == 'rewind_lt':
            if args.rewind_epoch:
                # learning rate rewinding
                for _ in range(args.rewind_epoch):
                    scheduler.step()
            start_epoch = args.rewind_epoch


def train(train_loader, model, criterion, optimizer, epoch):

    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    for i, (image, target) in enumerate(train_loader):

        if epoch < args.warmup:
            warmup_lr(epoch,
                      i + 1,
                      optimizer,
                      one_epoch_step=len(train_loader))

        image = image.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(image)
        loss = criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output, target)

        losses.update(loss.item(), image.size(0))
        top1.update(prec1, image.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Time {3:.2f}'.format(epoch,
                                        i,
                                        len(train_loader),
                                        end - start,
                                        loss=losses,
                                        top1=top1))
            start = time.time()

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, losses.avg


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (image, target) in enumerate(val_loader):

        image = image.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output, target)
        losses.update(loss.item(), image.size(0))
        top1.update(prec1, image.size(0))

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), loss=losses, top1=top1))

    print('valid_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def save_checkpoint(state,
                    is_SA_best,
                    save_path,
                    pruning,
                    filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_path, str(pruning) + filename)
    torch.save(state, filepath)
    if is_SA_best:
        shutil.copyfile(
            filepath,
            os.path.join(save_path,
                         str(pruning) + 'model_SA_best.pth.tar'))


def warmup_lr(epoch, step, optimizer, one_epoch_step):

    overall_steps = args.warmup * one_epoch_step
    current_steps = epoch * one_epoch_step + step

    lr = args.lr * current_steps / overall_steps
    lr = min(lr, args.lr)

    for p in optimizer.param_groups:
        p['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    # maxk = max(topk)
    batch_size = target.size(0)

    # _, pred = output.topk(maxk, 1, True, True)
    # pred = pred.t()
    pred = output.argmax(
        dim=1, keepdim=True)  # get the index of the max log-probability
    correct = pred.eq(target.view_as(pred)).sum().item()

    return correct * 100.0 / batch_size

    # res = []
    # for k in topk:
    # correct_k = correct[:k].view(-1).float().sum(0)
    # res.append(correct_k.mul_(100.0 / batch_size))
    # return res


def setup_seed(seed):
    print('setup random seed = {}'.format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
