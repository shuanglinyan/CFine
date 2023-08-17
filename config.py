import os
import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision.models as models
import torch.backends.cudnn as cudnn
import random
import numpy as np
import logging

from datasets.pedes import CuhkPedes
from models.model import Model
from utils import directory

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def data_config(image_dir, anno_dir, batch_size, split, max_length, transform, vocab_path='', min_word_count=0, cap_transform=None):
    
    data_split = CuhkPedes(image_dir, anno_dir, split, max_length, transform, \
                            vocab_path=vocab_path, min_word_count=min_word_count, cap_transform=cap_transform)
    if split == 'train':
        shuffle = True
        loader = data.DataLoader(data_split, batch_size, shuffle=shuffle, num_workers=4, drop_last=True)
    else:
        shuffle = False
        loader = data.DataLoader(data_split, batch_size, shuffle=shuffle, num_workers=4, drop_last=False)    
    return loader

def get_image_unique(image_dir, anno_dir, batch_size, split, max_length, transform):
    return CuhkPedes(image_dir, anno_dir, split, max_length, transform).unique

def network_config(args, split='train', param=None, resume=False, model_path=None, param2=None):
    network = Model(args)
    network = nn.DataParallel(network).cuda()
    cudnn.benchmark = True
    args.start_epoch = 0

    # process network params
    if resume:
        directory.check_file(model_path, 'model_file')
        checkpoint = torch.load(model_path)
        args.start_epoch = checkpoint['epoch'] + 1
        network.load_state_dict(checkpoint['network'])
        print('==> Loading checkpoint "{}"'.format(model_path))
    else:
        # pretrained
        if model_path is not None:
            print('==> Loading from pretrained models')
            network_dict = network.state_dict()
            # process keyword of pretrained model
            cnn_pretrained = torch.load(model_path)
            network_keys = network_dict.keys()
            prefix = 'module.image_model.'
            update_pretrained_dict = {}
            for k,v in cnn_pretrained.items():
                if prefix+k in network_keys:
                    update_pretrained_dict[prefix+k] = v
                if prefix+'branch2_'+k in network_keys:
                    update_pretrained_dict[prefix+'branch2_'+k] = v
                if prefix+'branch3_'+k in network_keys:
                    update_pretrained_dict[prefix+'branch3_'+k] = v
                if prefix+k not in network_keys and prefix+'branch2_'+k not in network_keys and prefix+'branch3_'+k not in network_keys:
                    print("warning: " + k + ' not load')
            network_dict.update(update_pretrained_dict)
            network.load_state_dict(network_dict)
                    
           
    # process optimizer params
    if split == 'test':
        optimizer = None
    else:
        # optimizer
        # different params for different part
        lang_params = list(map(id, network.module.language_model.parameters()))
        cnn_params = list(map(id, network.module.image_model.parameters()))
        cnn_params = cnn_params + lang_params
        other_params = filter(lambda p: id(p) not in cnn_params, network.parameters())
        other_params = list(other_params)

        if param is not None:
            other_params.extend(list(param))

        if param2 is not None:
            other_params.extend(list(param2))

        param_groups = [{'params':other_params},
            {'params':network.module.image_model.parameters(), 'weight_decay':args.wd, 'lr':args.lr/10},
            {'params':network.module.language_model.parameters(), 'lr':args.lr/10}]
        optimizer = torch.optim.Adam(
            param_groups,
            lr = args.lr, betas=(args.adam_alpha, args.adam_beta), eps=args.epsilon)
        if resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
    logger.info('Total params: %2.fM' % (sum(p.numel() for p in network.parameters()) / 1000000.0))

    return network, optimizer


def log_config(args, ca):
    filename = args.checkpoint_dir +'/' + ca + '.txt'
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s : %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(handler)
    for k in args.__dict__:
        logger.info(str(k)+' : '+str(args.__dict__[k]))


def dir_config(args):
    if not os.path.exists(args.image_dir):
        raise ValueError('Supply the dataset directory with --image_dir')
    if not os.path.exists(args.anno_dir):
        raise ValueError('Supply the anno file with --anno_dir')
    # save checkpoint
    directory.makedir(args.checkpoint_dir)


def lr_scheduler(optimizer, args):
    if '_' in args.epoches_decay:
        epoches_list = args.epoches_decay.split('_')
        epoches_list = [int(e) for e in epoches_list]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, epoches_list)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(args.epoches_decay))
    return scheduler
