import collections.abc
import logging
import os
import pathlib

import git
import numpy as np
import yaml
from omegaconf import ListConfig, OmegaConf


def load_auto_loading(cfg):
    """
    Loads automatically the yaml files described under the key "auto_loading" to the root node
    """
    if 'auto_loading' not in cfg.keys():
        # No auto loading has been  defined
        return
    a_cfg = cfg.auto_loading
    #! Iteration for each path defined in cfg.auto_loading
    for key, path_cfg in a_cfg.items():
        with open(path_cfg, "r") as f:
            _cfg = yaml.safe_load(f)
        cfg[key] = _cfg


def short_decode_id(list_keys, *, _parent_):
    """
    Decodes (in a short way) a set of keys as a unique string
    """
    decoded_id = ''
    for key in list_keys:
        path_keys = key.split(".")
        __nested_value = _parent_
        for _key in path_keys:
            __nested_value = __nested_value[_key]
        assert not isinstance(__nested_value, dict)
        decoded_id += f"{path_keys[-1]}.{__nested_value}_"
    return decoded_id


def decode_id(list_keys, *, _parent_):
    """
    Decodes (in a short way) a set of keys as a unique string
    """
    decoded_id = ''
    for key in list_keys:
        path_keys = key.split(".")
        __nested_value = _parent_
        for _key in path_keys:
            __nested_value = __nested_value[_key]
        assert not isinstance(__nested_value, dict)

        decoded_id += f"{key}.{__nested_value}_"
    return decoded_id


def rel_path(rel_path, *, _parent_):
    path = pathlib.Path(os.path.join(CFG_ROOT, rel_path)).resolve()
    return path.__str__()


def encode_list(key_list, *, _parent_):
    data_list = _parent_[key_list]
    data = ""
    for dt in [item for sublist in data_list for item in sublist]:
        data += f"{dt}_"
    return data[:-1]


def range(range_info, *, _parent_):
    np_range = np.arange(range_info[0], range_info[1], range_info[2])
    return ListConfig([float(n) for n in np_range])


def linspace(range_info, *, _parent_):
    np_range = np.linspace(range_info[0], range_info[1], range_info[2])
    return ListConfig([float(n) for n in np_range])


OmegaConf.register_new_resolver('decode', decode_id)
OmegaConf.register_new_resolver('short_decode', short_decode_id)
OmegaConf.register_new_resolver('rel_path', rel_path)
OmegaConf.register_new_resolver('encode_list', encode_list)
OmegaConf.register_new_resolver('range', range)
OmegaConf.register_new_resolver('linspace', linspace)


def get_default(cfg):
    df = OmegaConf.create(dict(default_cfg=cfg["default_cfg"]))
    OmegaConf.resolve(df)
    assert os.path.exists(df.default_cfg)
    # ! Reading YAML file

    with open(df.default_cfg, "r") as f:
        cfg_dict = yaml.safe_load(f)
    logging.info(f"Loaded default cfg from: {df.default_cfg}")
    return cfg_dict


def add_git_commit(cfg):
    repo = git.Repo(search_parent_directories=True)
    cfg['git_commit'] = repo.head._get_commit().name_rev
    return cfg


def update_cfg(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_cfg(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def set_loggings():    
    logging.basicConfig(
        format='[%(levelname)s] [%(asctime)s]:  %(message)s',
        level=logging.INFO
    )

def read_omega_cfg(cfg_file):
    assert os.path.exists(cfg_file), f"File does not exist {cfg_file}"

    # ! Reading YAML file
    with open(cfg_file, "r") as f:
        cfg_dict = yaml.safe_load(f)

    set_loggings()
    # ! Saving cfg root for relative_paths
    global CFG_ROOT
    CFG_ROOT = os.path.dirname(cfg_file)

    # ! add git commit
    cfg_dict = add_git_commit(cfg_dict)
    if "default_cfg" in cfg_dict.keys():
        df = get_default(cfg_dict)
        update_cfg(df, cfg_dict)
        cfg = OmegaConf.create(df)
    else:
        cfg = OmegaConf.create(cfg_dict)

    # ! Loading Auto-loadings
    load_auto_loading(cfg)
    return cfg


def get_empty_cfg():
    logging.basicConfig(
        format='[%(levelname)s] [%(asctime)s]:  %(message)s',
        level=logging.INFO
    )
    cfg_dict= dict()
    
    # ! add git commit
    cfg_dict = add_git_commit(cfg_dict)

    cfg = OmegaConf.create(cfg_dict)
    return cfg

def read_config(cfg_file):
    cfg = read_omega_cfg(cfg_file)
    OmegaConf.resolve(cfg)
    cfg = set_cfg(cfg)
    logging.info(f"Config file loaded successfully from: {cfg_file}")
    return cfg


def set_cfg(cfg):
    OmegaConf.set_readonly(cfg, True)
    OmegaConf.set_struct(cfg, True)
    return cfg


def save_cfg(cfg_file, cfg):
    with open(cfg_file, 'w') as fn:
        OmegaConf.save(config=cfg, f=fn)
