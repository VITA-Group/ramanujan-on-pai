from __future__ import print_function

import argparse
import collections
import copy
import hashlib
import json
import logging
import math
import os
import time
import warnings

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import lib.sparselearning
from lib import prune
from lib.common_models.models import models as MODELS
from lib.common_models.utils import add_log_softmax
from lib.common_models.utils import stripping_bias
from lib.common_models.utils import stripping_skip_connections
from lib.sparselearning.utils import get_cifar100_dataloaders
from lib.sparselearning.utils import get_cifar10_dataloaders
from lib.sparselearning.utils import get_mnist_dataloaders
from lib.sparselearning.utils import plot_class_feature_histograms

warnings.filterwarnings("ignore", category=UserWarning)
cudnn.benchmark = True
cudnn.deterministic = True

if not os.path.exists('./models'): os.mkdir('./models')
if not os.path.exists('./logs'): os.mkdir('./logs')
logger = None

models = {}
for name, fn in MODELS.items():
    models[name] = (fn, [])


def prune_loop(model,
               loss,
               pruner,
               dataloader,
               device,
               density,
               scope,
               epochs,
               train_mode=False):

    # Set model to train or eval mode
    model.train()
    if not train_mode:
        model.eval()

    # Prune model
    for epoch in tqdm(range(epochs)):
        pruner.score(model, loss, dataloader, device)
        sparse = 1 - density**((epoch + 1) / epochs)
        pruner.mask(sparse, scope)


def get_sparse_model(args, model, loss, dataloader, device, save_dir):
    prunemethod = args.prune
    if args.sparse:
        if args.prune in ('SynFlow', 'iterSNIP'):
            iteration = 100
            if args.prune == 'iterSNIP':
                prunemethod = 'SNIP'
        else:
            iteration = 1
        pruner = eval(f"prune.{prunemethod}")(prune.masked_parameters(model))
        if not args.retrain_mask:
            prune_loop(model, loss, pruner, dataloader, device, args.density,
                       'global', iteration)
        else:
            list(prune.masked_parameters(model))
            model.load_state_dict(
                torch.load(os.path.join(save_dir, 'init.pth.tar'),
                           map_location='cpu'))
        prune.check_sparsity(model)
        return model
    else:
        args.density = 1.0
        args.prune = "none"
        return model


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def setup_logger(args):
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)

    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    args_copy.seed = 0

    log_path = './logs/{0}_{1}_{2}.log'.format(
        args.model, args.density,
        hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s',
                                  datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg)


def train(args, model, device, train_loader, optimizer, epoch, mask=None):
    model.train()
    train_loss = 0
    correct = 0
    n = 0
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        if args.fp16: data = data.half()
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        train_loss += loss.item()
        pred = output.argmax(
            dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        n += target.shape[0]

        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print_and_log(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {}/{} ({:.3f}% '
                .format(epoch, batch_idx * len(data),
                        len(train_loader) * args.batch_size,
                        100. * batch_idx / len(train_loader), loss.item(),
                        correct, n, 100. * correct / float(n)))

    # training summary
    print_and_log(
        '\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            'Training summary', train_loss / batch_idx, correct, n,
            100. * correct / float(n)))
    return train_loss / batch_idx


def evaluate(args, model, device, test_loader, is_test_set=False):
    model.eval()
    test_loss = 0
    correct = 0
    n = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if args.fp16: data = data.half()
            # model.t = target
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(
                dim=1,
                keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]

    test_loss /= float(n)

    print_and_log(
        '\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            'Test evaluation' if is_test_set else 'Evaluation', test_loss,
            correct, n, 100. * correct / float(n)))
    return correct / float(n)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size',
                        type=int,
                        default=256,
                        metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size',
                        type=int,
                        default=128,
                        metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--multiplier',
                        type=int,
                        default=1,
                        metavar='N',
                        help='extend training time by multiplier times')
    parser.add_argument('--epochs',
                        type=int,
                        default=250,
                        metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr',
                        type=float,
                        default=0.1,
                        metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed',
                        type=int,
                        default=17,
                        metavar='S',
                        help='random seed (default: 17)')
    parser.add_argument('--save_dir',
                        type=str,
                        default='results',
                        help='main saving dir')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        metavar='N',
        help='how many batches to wait before logging training status')
    parser.add_argument(
        '--optimizer',
        type=str,
        default='sgd',
        help='The optimizer to use. Default: sgd. Options: sgd, adam.')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--l1', type=float, default=0.0)
    parser.add_argument('--fp16',
                        action='store_true',
                        help='Run in fp16 mode.')
    parser.add_argument('--valid_split', type=float, default=0.1)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--l2', type=float, default=5e-4)
    parser.add_argument('--retrain_mask', action='store_true')
    parser.add_argument(
        '--iters',
        type=int,
        default=1,
        help=
        'How many times the model should be run after each other. Default=1')
    parser.add_argument(
        '--save-features',
        action='store_true',
        help=
        'Resumes a saved model and saves its feature data to disk for plotting.'
    )
    parser.add_argument(
        '--bench',
        action='store_true',
        help='Enables the benchmarking of layers and estimates sparse speedups'
    )
    parser.add_argument('--max-threads',
                        type=int,
                        default=10,
                        help='How many threads to use for data loading.')
    parser.add_argument(
        '--decay-schedule',
        type=str,
        default='cosine',
        help=
        'The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.'
    )
    parser.add_argument('--nolrsche',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')
    parser.add_argument('--no_rewire_extend',
                        action='store_true',
                        default=False,
                        help='if ture, only do rewire for 250 epoochs')
    parser.add_argument('-j',
                        '--workers',
                        default=10,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 10)')
    parser.add_argument('--world-size',
                        default=-1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--mgpu',
                        action='store_true',
                        help='Enable snip initialization. Default: True.')
    parser.add_argument('--prune', type=str, default='ERK')
    parser.add_argument('--density',
                        type=float,
                        default=0.05,
                        help='The density of the overall sparse network.')
    parser.add_argument('--sparse',
                        action='store_true',
                        help='Enable sparse mode. Default: True.')
    parser.add_argument('--fix',
                        action='store_true',
                        help='Fix topology during training. Default: True.')
    parser.add_argument(
        '--strip-bias',
        action='store_true',
    )
    parser.add_argument(
        '--strip-skip',
        action='store_true',
    )

    args = parser.parse_args()
    setup_logger(args)
    print_and_log(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'device will be chosen as {device} for this run.')

    print_and_log('\n\n')
    print_and_log('=' * 80)
    torch.manual_seed(args.seed)
    for i in range(args.iters):
        print_and_log("\nIteration start: {0}/{1}\n".format(i + 1, args.iters))
        outputs = 10
        if args.data == 'cifar10':
            train_loader, valid_loader, test_loader = get_cifar10_dataloaders(
                args, args.valid_split, max_threads=args.max_threads)
            outputs = 10
        elif args.data == 'cifar100':
            train_loader, valid_loader, test_loader = get_cifar100_dataloaders(
                args, args.valid_split, max_threads=args.max_threads)
            outputs = 100
        else:
            raise NotImplementedError

        if args.model not in models:
            print(
                'You need to select an existing model via the --model argument. Available models include: '
            )
            for key in models:
                print('\t{0}'.format(key))
            raise Exception('You need to select a model')
        else:
            cls, cls_args = models[args.model]
            model = cls(num_classes=outputs,
                        seed=args.seed,
                        strip_bias=args.strip_bias,
                        strip_skip=args.strip_skip).cuda()
            add_log_softmax(model)

        if args.mgpu:
            print('Using multi gpus')
            model = torch.nn.DataParallel(model).to(device)

        timestr = time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
        if not args.retrain_mask:
            save_dir = os.path.join(args.save_dir, f"density_{args.density}",
                                    args.data, args.model, args.prune,
                                    str(args.seed), timestr)
        else:
            save_dir = os.path.join(args.save_dir, f"density_{args.density}",
                                    args.data, args.model, args.prune,
                                    str(args.seed), 'latest')

        os.makedirs(save_dir, exist_ok=True)

        model = get_sparse_model(args, model, F.nll_loss, train_loader, device,
                                 save_dir)
        optimizer = None
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(),
                                  lr=args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.l2)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(),
                                   lr=args.lr,
                                   weight_decay=args.l2)
        else:
            print('Unknown optimizer: {0}'.format(args.optimizer))
            raise Exception('Unknown optimizer.')

        if args.nolrsche:
            lr_scheduler = None
        else:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[int(args.epochs / 2),
                            int(args.epochs * 3 / 4)],
                last_epoch=-1)
        if args.resume:
            if os.path.isfile(args.resume):
                print_and_log("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print_and_log("=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint['epoch']))
                print_and_log('Testing...')
                evaluate(args, model, device, test_loader)
                model.feats = []
                model.densities = []
                plot_class_feature_histograms(args, model, device,
                                              train_loader, optimizer)
            else:
                print_and_log("=> no checkpoint found at '{}'".format(
                    args.resume))

        mask = None

        with open(os.path.join(save_dir, 'config.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        # save_checkpoint(model.state_dict(),
        # os.path.join(save_dir, "init.pth.tar"))
        best_acc = 0.0
        acc = collections.defaultdict(list)

        for epoch in range(1, args.epochs + 1):

            t0 = time.time()
            loss = train(args, model, device, train_loader, optimizer, epoch,
                         mask)

            if lr_scheduler: lr_scheduler.step()

            if args.valid_split > 0.0:
                val_acc = evaluate(args, model, device, valid_loader)
                test_acc = evaluate(args, model, device, test_loader, True)
                acc['val_acc'].append(val_acc)
                acc['test_acc'].append(test_acc)
                acc['epoch'].append(epoch)
            acc['train_loss'].append(loss)

            if val_acc > best_acc:
                print('Saving model')
                best_acc = val_acc
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                    },
                    filename=os.path.join(save_dir, 'model.pth'))

            print_and_log(
                'Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n'
                .format(optimizer.param_groups[0]['lr'],
                        time.time() - t0))
            if math.isnan(loss):
                print("bad mask! leaving early!")
                break

        if len(acc['epoch']) < args.epochs:
            pad = [float('nan')] * (args.epochs - len(acc['epoch']))
            acc['val_acc'].extend([acc['val_acc'][-1]] *
                                  (args.epochs - len(acc['epoch'])))
            acc['test_acc'].extend([acc['test_acc'][-1]] *
                                   (args.epochs - len(acc['epoch'])))
            acc['train_loss'].extend([acc['train_loss'][-1]] *
                                     (args.epochs - len(acc['epoch'])))
            acc['epoch'].extend(
                list(range(acc['epoch'][-1] + 1, args.epochs + 1)))

        df = pd.DataFrame.from_dict(acc)
        df.to_csv(os.path.join(save_dir, "results.csv"))
        print_and_log("\nIteration end: {0}/{1}\n".format(i + 1, args.iters))


if __name__ == '__main__':
    main()
