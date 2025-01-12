# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 13:11:56 2024

@author: robur
"""

from dataLoader import SinWaveDataset
from GANModels import Generator, Discriminator
from functions import train, save_samples_plots, LinearLrDecay, load_params, copy_params
from utils.utils import set_log_dir, save_checkpoint, load_config
import torch
import torch.utils.data.distributed
from torch.utils import data
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy



def load_data(cfg):
    """
    Function to load the input data and create the data loader
    object for bath processing.
    """
    loader = SinWaveDataset(normalize=False)
    train_set, test_set = loader.loadData()
    train_loader = data.DataLoader(train_set, batch_size=cfg['batch_size'], num_workers=0, shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=cfg['batch_size'], num_workers=0, shuffle=True)
    return train_loader, test_loader


def initialize_models(cfg):
    """
    Initialize the architecture of both the generator and the discriminator model
    and load the models to GPU (if available).
    """
    gen_net = Generator(seq_len=cfg['gen_arch']['seq_len'],
                        channels=cfg['gen_arch']['channels'],
                        patch_size=cfg['gen_arch']['patch_size'],
                        latent_dim=cfg['gen_arch']['latent_dim'],
                        embed_dim=cfg['gen_arch']['embed_dim'],
                        depth=cfg['gen_arch']['depth'],
                        num_heads=cfg['gen_arch']['num_heads'],
                        forward_expansion=cfg['gen_arch']['forward_expansion'],
                        forward_drop_rate=cfg['gen_arch']['forward_drop_rate'],
                        attn_drop_rate=cfg['gen_arch']['attn_drop_rate'])
    
    dis_net = Discriminator(seq_length=cfg['dis_arch']['seq_length'], 
                            in_channels=cfg['dis_arch']['in_channels'],
                            patch_size=cfg['dis_arch']['patch_size'],
                            emb_size=cfg['dis_arch']['emb_size'], 
                            depth=cfg['dis_arch']['depth'],
                            num_heads=cfg['dis_arch']['num_heads'],
                            drop_p=cfg['dis_arch']['drop_p'],
                            forward_expansion=cfg['dis_arch']['forward_expansion'],
                            forward_drop_p=cfg['dis_arch']['forward_drop_p'],
                            n_classes=cfg['dis_arch']['n_classes'])

    if torch.cuda.is_available():
        gen_net.cuda()
        dis_net.cuda()
    else:
        print('using CPU, this will be slow')

    return gen_net, dis_net


def initialize_optimizers(cfg, gen_net, dis_net):
    """
    Initilaization of training optimizer and
    learning rate schedualar for both generator
    and discriminator models.
    """
    if cfg['optimizer'] == "adam":
        gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                         cfg['g_lr'], (cfg['beta1'], cfg['beta2']))
        dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                         cfg['d_lr'], (cfg['beta1'], cfg['beta2']))
    elif cfg['optimizer'] == "adamw":
        gen_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                          cfg['g_lr'], weight_decay=cfg['wd'])
        dis_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                          cfg['d_lr'], weight_decay=cfg['wd'])

    gen_scheduler = LinearLrDecay(gen_optimizer, cfg['g_lr'], 0.0, 0, cfg['max_iter'] * cfg['n_critic'] * len(train_loader))
    dis_scheduler = LinearLrDecay(dis_optimizer, cfg['d_lr'], 0.0, 0, cfg['max_iter'] * cfg['n_critic'] * len(train_loader))

    return gen_optimizer, dis_optimizer, gen_scheduler, dis_scheduler


def initialize_training_state(cfg, gen_net):
    """
    Initialization of training state, laten sapce.
    """
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (100, cfg['latent_dim'])))
    avg_gen_net = deepcopy(gen_net).cpu()
    gen_avg_param = copy_params(avg_gen_net)
    del avg_gen_net

    cfg['path_helper'] = set_log_dir('logs', cfg['exp_name'])
    writer = SummaryWriter(cfg['path_helper']['log_path'])

    writer_dict = {
        'writer': writer,
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    return fixed_z, gen_avg_param, writer_dict


def train_loop(cfg, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, fixed_z, writer_dict, gen_scheduler, dis_scheduler):
    """
    Training loop function.
    """
    for epoch in range(cfg['max_epoch']):
        print(f"Epoch {epoch}")
        lr_schedulers = (gen_scheduler, dis_scheduler) if cfg['lr_decay'] else None
        train(cfg, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict, fixed_z, lr_schedulers)

        if cfg['show'] and epoch % cfg['val_freq'] == 0:
            backup_param = copy_params(gen_net)
            load_params(gen_net, gen_avg_param, mode="cpu")
            save_samples_plots(cfg, fixed_z, epoch, gen_net)
            load_params(gen_net, backup_param, mode='gpu')

            avg_gen_net = deepcopy(gen_net)
            load_params(avg_gen_net, gen_avg_param)

            save_checkpoint({
                'cfg': cfg,
                'epoch': epoch,
                'gen_state_dict': gen_net.state_dict(),
                'dis_state_dict': dis_net.state_dict(),
                'avg_gen_state_dict': avg_gen_net.state_dict(),
                'gen_optimizer': gen_optimizer.state_dict(),
                'dis_optimizer': dis_optimizer.state_dict(),
                'best_fid': 1e4,
                'path_helper': cfg['path_helper'],
                'fixed_z': fixed_z
            }, False, cfg['path_helper']['ckpt_path'], filename=f"checkpoint_{epoch}.pth")


if __name__ == "__main__":
    cfg = load_config('cfg.json')
    train_loader, test_loader = load_data(cfg)
    gen_net, dis_net = initialize_models(cfg)
    gen_optimizer, dis_optimizer, gen_scheduler, dis_scheduler = initialize_optimizers(cfg, gen_net, dis_net)
    fixed_z, gen_avg_param, writer_dict = initialize_training_state(cfg, gen_net)

    cfg['max_epoch'] *= cfg['n_critic']
    train_loop(cfg, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, fixed_z, writer_dict, gen_scheduler, dis_scheduler)

