'''Train CIFAR10/CIFAR100 with PyTorch'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import argparse
import numpy as np
import random 
import time 
import logging
import sys 
import warnings
import os 

import models.cifar as models 

from utils.utils import ProgressMeter, AverageMeter, save_checkpoint, accuracy
from utils.ptflops import get_model_complexity_info
from utils.dataset import prepare_test_data, prepare_train_data
from models.modules.qconv import qconv_type


conv_types = ['fp', 'cpt']

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 QuanTraining')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='cifar10_resnet20',
                        help='model architecture (default: resnet18)')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_schedule', default='piecewise', type=str,
                        help='learning rate schedule')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--evaluate', default=False, action='store_true', 
                        help='evaluate for model')
    parser.add_argument('--save', type=str, default='EXP', 
                        help='path for saving trained models')
    parser.add_argument('--seed', default=False, action='store_true', 
                        help='set random seed for training')
    parser.add_argument('--manual-seed', default=2, type=int, help='random seed is settled')
    parser.add_argument('-g', '--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('-p', '--print-freq', default=50, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')
    parser.add_argument('-d', '--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--epochs', default=100, type=int, help='training epochs')

    parser.add_argument('--actbits', default=0, type=int, help='bit-width for activations')
    parser.add_argument('--wbits', default=0, type=int, help='bit-width for weights')
    parser.add_argument('--gbits', default=0, type=0, help='bit-width for gradients')
    
    parser.add_argument('--conv_type', default='fp', type=str, help='the type of convolutions')

    args = parser.parse_args()
    return args


def main(): 
    args = parse_args()
    if args.seed:
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        np.random.seed(args.manual_seed)
        random.seed(args.manual_seed)  # 设置随机种子

    args.save = 'train-{}-{}-{}'.format(args.arch, args.save, time.strftime("%Y%m%d-%H%M%S"))

    from tensorboardX import SummaryWriter
    writer_comment = args.save 
    log_dir = '{}/{}'.format('logger', args.save)
    writer = SummaryWriter(log_dir = log_dir, comment=writer_comment)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')


    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    logging.info('==> Preparing data..')

    trainloader = prepare_train_data(args.dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    testloader = prepare_test_data(args.dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    logging.info('==> Building model..')

    model = models.__dict__[args.arch](args.conv_type)

    model = model.to(device)

    if device == 'cuda': 
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True 
    
    if args.resume: 
        logging.info('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/{}-ckpt.pth'.format(args.arch))
        model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    
    scheduler = torch.optim.lr_scheduler.ConsineAnnealingLR(optimizer, T_max=args.epochs)


    if args.evaluate:
        validate(testloader, model, criterion, args)
    
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_top1 = train(trainloader, model, criterion, optimizer, epoch, args)
        validate_loss, validate_top1 = validate(testloader, model, criterion, args)
        logging.info('the train loss is : {} ; For Train !  the top1 accuracy is : {} '.format(train_loss, train_top1))
        logging.info('the validate loss is : {} ; For Validate !  the top1 accuracy is : {} '.format(validate_loss, validate_top1))
        writer.add_scalars('Train-Loss/Training-Validate',{
            'train_loss': train_loss,
            'validate_loss': validate_loss
        }, epoch + 1)
        writer.add_scalars('Train-Top1/Training-Validate',{
            'train_acc1': train_top1,
            'validate_acc1': validate_top1
        }, epoch + 1)
        writer.add_scalar('Learning-Rate-For-Train', 
            optimizer.state_dict()['param_groups'][0]['lr'],
            epoch + 1)

        if validate_top1 > best_acc:
            best_acc = validate_top1
            logging.info('the best model top1 is : {} and its epoch is {} !'.format(best_acc, epoch))
            state = {
                'net': model.module.state_dict(),
                'acc': best_acc, 
                'epoch': epoch
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/{}-ckpt.pth'.format(args.arch))
        scheduler.step()


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                            prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images, args.actbits, args.wbits, args.gbits)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1,  = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)

    return losses.avg, top1.avg

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images, args.actbits, args.wbits, args.gbits)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1,  = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        logging.info(' * Acc@1 {top1.avg:.3f} '
              .format(top1=top1))

    return losses.avg, top1.avg