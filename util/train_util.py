import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
import os

from util.utils import save_checkpoint, log
from util.utils import tiny_imagenet, CIFAR10Imbalanced
from util.test import test_all_datasets
from criterion import NTXent
from torch.utils.data import DataLoader
import json
from criterion.focal import BalancedFocalLoss 
from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageFolder
from util.test import train_or_val
from augmentations import SimSiamTransform
from models import LinearEvaluation
from util.curr_learning import ClassProbabilitySampler
from util.test import testSSL, test_currSSL


def get_criteria(args):
    """
    Loss criterion / criteria selection for training
    """
    criteria = {
        # 'ntxent': [NTXent(args), args.train.criterion_weight[0]]
        'focal': [BalancedFocalLoss(args), args.train.criterion_weight[0]]
    }

    return criteria


def get_proba():
    # Open the text file in read mode
    with open('convert.txt', 'r') as file:
        # Read the contents of the file
        content = file.read()

        # Parse the content as a dictionary
        dictionary = json.loads(content)

    # Print the dictionary
    return dictionary['1'][0]

def trainloaderSSL_Curr(args, transform, class_probas, epoch, imagenet_split='train'):

    # class_probas = get_proba()
    # class_probas = [1-i for i in class_probas]
    if args.train.dataset.name == 'CIFAR100':
        train_dataset = CIFAR100(args.train.dataset.data_dir, train=True, download=True, transform=transform)
    elif args.train.dataset.name == 'ImageNet':
        train_dataset = ImageFolder(os.path.join(args.train.dataset.data_dir, imagenet_split), transform=transform)
    elif args.train.dataset.name == 'CIFAR10':
        train_dataset = CIFAR10(args.train.dataset.data_dir, train=True, download=True, transform=transform)
    elif args.train.dataset.name == 'CIFAR10Imbalanced':
        train_dataset = CIFAR10Imbalanced(root=args.train.dataset.data_dir, train=True, download=True, transform=transform)
    elif args.train.dataset.name == 'TinyImageNet':
        train_dataset = tiny_imagenet(args.train.dataset.data_dir, train=True, transform=transform)
    elif args.train.dataset.name == 'STL10':
        train_dataset = STL10(args.train.dataset.data_dir, split="unlabeled", download=True, transform=transform)
        
    class_probas = [0.2 if x != x else x for x in class_probas]
    if epoch %20==0:
        class_probas = [x+0.3 if x <0.2 else x for x in class_probas]
    elif epoch %50==0:
        class_probas = [x+0.5 if x <0.4 else x for x in class_probas]
    elif epoch %80==0:
        class_probas = [x+0.6 if x <0.3 else x for x in class_probas]
    # log(class_probas)
    # class_probas[4] = 0.2
    sampler = ClassProbabilitySampler(train_dataset, class_probas)

    log('sample finished')
    train_loader = DataLoader(train_dataset, batch_size=args.train.batchsize, sampler=sampler, drop_last=True, num_workers=args.train.num_workers)
    log('data loaded')
    return train_loader


def write_scalar(writer, total_loss, loss_p_c, leng, epoch):
    """
    Add Loss scalars to tensorboard
    """
    writer.add_scalar("Total_Loss/train", total_loss / leng, epoch)
    for k in loss_p_c:
        writer.add_scalar("{}_Loss/train".format(k), loss_p_c[k] / leng, epoch)

def multi_acc(pred, label):
  accs_per_label_pct = []
  tags = torch.argmax(pred, dim=1)
  for c in range(3):  # the three classes
    of_c = label == c
    num_total_per_label = of_c.sum()
    of_c &= tags == label
    num_corrects_per_label = of_c.sum()
    accs_per_label_pct.append(num_corrects_per_label / num_total_per_label * 100)
  return accs_per_label_pct


def train_one_epoch(args, train_loader, model, criteria, optimizer, scheduler, epoch):
    """
    Train one epoch of SSL model
    """
    loss_per_criterion = {}
    total_loss = 0
    for i, ((x1, y1), targets) in enumerate(train_loader):
        x1 = x1.cuda(device=args.device)
        y1 = y1.cuda(device=args.device)
        optimizer.zero_grad()

        if args.train.model == 'simclr':
            _, _, zx, zy = model(x1, y1)
        elif args.train.model == 'simsiam':
            fx, fy, zx, zy, px, py = model(x1, y1)

        loss = torch.tensor(0).to(args.device)
        for k in criteria:
            if k == 'ntxent':
                criterion_loss = criteria[k][0](zx, zy)
            if k == 'focal':
                criterion_loss = criteria[k][0](zx, zy)
            elif k == 'simsiam':
                criterion_loss = criteria[k][0](zx, zy, px, py)
            if k not in loss_per_criterion:
                loss_per_criterion[k] = criterion_loss
            else:
                loss_per_criterion[k] += criterion_loss
            loss = torch.add(loss, torch.mul(criterion_loss, criteria[k][1]))
            
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if i % 50 == 0:
            log("Batch {}/{}. Loss: {}.  Time elapsed: {} ".format(i, len(train_loader), loss.item(),
                                                                   datetime.now() - args.start_time))
        total_loss += loss.item()
    return total_loss, loss_per_criterion


def trainSSL(args, model, train_loader, optimizer, criteria, writer, scheduler=None):
    """
    Train a SSL model
    """
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        log('Model converted to DP model with {} cuda devices'.format(torch.cuda.device_count()))
    model = model.to(args.device)

    for epoch in tqdm(range(1, args.train.epochs + 1)):
        model = model.to(args.device)
        model.train()
        total_loss, loss_per_criterion = train_one_epoch(args, train_loader, model, criteria, optimizer, scheduler, epoch)
        log("Epoch {}/{}. Total Loss: {}.   Time elapsed: {} ".
            format(epoch, args.train.epochs, total_loss / len(train_loader), datetime.now() - args.start_time))

        write_scalar(writer, total_loss, loss_per_criterion, len(train_loader), epoch)

        # Save the model at specific checkpoints
        if epoch % 10 == 0:
            if torch.cuda.device_count() > 1:
                save_checkpoint(state_dict=model.module.state_dict(), args=args, epoch=epoch,
                                filename='checkpoint_model_{}.pth'.format(epoch))
            else:
                save_checkpoint(state_dict=model.state_dict(), args=args, epoch=epoch,
                                filename='checkpoint_model_{}.pth'.format(epoch))
        if epoch %20 ==0:
            transform = SimSiamTransform(args.train.dataset.img_size)

            test_currSSL(args, writer, model, multi = True)

            result_list = get_proba()
            train_loader = trainloaderSSL_Curr(args, transform, result_list, epoch, imagenet_split='train')

    log("Total training time {}".format(datetime.now() - args.start_time))

    # Test the SSl Model
    if torch.cuda.device_count() > 1:
        test_all_datasets(args, writer, model.module)
    else:
        test_all_datasets(args, writer, model)

    writer.close()
