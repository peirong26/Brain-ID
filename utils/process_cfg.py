"""Wrapper to train/test models."""

import os
import pytz
from datetime import datetime

from utils.config import Config

def update_config(cfg, exp_name='', job_name=''):
    """
    Update some configs.
    Args:
        cfg: <Config> from submit_config.config
    """
    tz_NY = pytz.timezone('America/New_York')

    if 'lemon' in cfg.out_root:
        cfg.out_dir = os.path.join(cfg.root_dir_lemon, cfg.out_dir) 
    else:
        cfg.out_dir = os.path.join(cfg.root_dir_yogurt_out, cfg.out_dir)

    cfg.vis_itr = int(cfg.vis_itr / cfg.num_gpus)


    if cfg.eval_only:
        cfg.out_dir = os.path.join(cfg.out_dir, 'Test', exp_name, job_name, datetime.now(tz_NY).strftime("%m%d-%H%M"))
    else:
        cfg.out_dir = os.path.join(cfg.out_dir, exp_name, job_name, datetime.now(tz_NY).strftime("%m%d-%H%M"))
    return cfg


def merge_and_update_from_dict(cfg, dct):
    """
    (Compatible for submitit's Dict as attribute trick)
    Merge dict as dict() to config as CfgNode().
    Args:
        cfg: dict
        dct: dict
    """
    if dct is not None:
        for key, value in dct.items():
            if isinstance(value, dict):
                if key in cfg.keys():
                    sub_cfgnode = cfg[key]
                else:
                    sub_cfgnode = dict()
                    cfg.__setattr__(key, sub_cfgnode) 
                sub_cfgnode = merge_and_update_from_dict(sub_cfgnode, value)
            else:
                cfg[key] = value
    return cfg


def load_config(default_cfg_file, add_cfg_files = [], cfg_dir = ''):
    cfg = Config(default_cfg_file) 
    for cfg_file in add_cfg_files:
        if os.path.isabs(cfg_file):
            add_cfg = Config(cfg_file)
        else:
            assert os.path.isabs(cfg_dir)
            if not cfg_file.endswith('.yaml'):
                cfg_file += '.yaml'
            add_cfg = Config(os.path.join(cfg_dir, cfg_file))
        cfg = merge_and_update_from_dict(cfg, add_cfg)
    return update_config(cfg, exp_name=cfg["exp_name"], job_name = cfg["job_name"])
    
