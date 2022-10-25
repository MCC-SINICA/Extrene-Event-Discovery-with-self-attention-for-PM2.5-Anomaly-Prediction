import os, shutil
import numpy as np
import random
import argparse
from argparse import Namespace
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from model import *
from networks import *
from custom_loss import *
import csv
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def check_train_id(opt):
    assert opt.no != None, f"no should be a number"

def build_dirs(opt):
    try:
        cfg_dir = os.makedirs(os.path.join(opt.cfg_dir), 0o777)
    except:
        pass
    cpt_dir = os.path.join(opt.cpt_dir, str(opt.no))
    log_dir = os.path.join(opt.log_dir, str(opt.no))

    if (not opt.yes) and os.path.exists(cpt_dir):
        res = input(f"no: {no} exists, are you sure continue training? It will override all files.[y:N]")
        res = res.lower()
        assert res in ["y", "yes"], "Stop training!!"
        print("Override all files.")

    if not os.path.exists(cpt_dir):
        os.makedirs(cpt_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

def write_record(path, records):
    header = ["sitename", "mode", "best_rmse", "epoch", "cost_time"]
    with open(path, "w") as fp:
        writer = csv.DictWriter(fp, fieldnames=header)
        writer.writeheader()
        for sitename in records:
            writer.writerow({
                "sitename": sitename,
                "mode":      records[sitename]["mode"],
                "best_rmse": records[sitename]["best_rmse"],
                "epoch":     records[sitename]["epoch"],
                "cost_time": records[sitename]["timestamp"]
            })

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

#################################
########## config ###############
#################################
def parse(args=None, data_version=1):
    parser = argparse.ArgumentParser()
    try: 
        if data_version == 1:
            from argument import add_arguments  #argument.py
        elif data_version ==2:
            from argument2 import add_arguments
        else:
            print('data_version error ! check train.py and utils parse')
        parser = add_arguments(parser)
    except:
        pass 
    if args is not None:
        config = parser.parse_args(args=args)
    else:
        config = parser.parse_args()
    config = update_config(config)
    return config

def parse_version(args=None):
    parser = argparse.ArgumentParser()
    try: 
        from argument_version import add_arguments
        parser = add_arguments(parser)
    except:
        pass 
    
    if args is not None:
        config = parser.parse_args(args=args)
    else:
        config = parser.parse_args()
    config = update_config(config)
    config.origin_all_dir=f"total_dataset/data_{config.ratio}/origin/all"
    config.origin_train_dir=f"total_dataset/data_{config.ratio}/origin/train"
    config.origin_valid_dir=f"total_dataset/data_{config.ratio}/origin/valid"
    config.norm_train_dir=f"total_dataset/data_{config.ratio}/norm/train"
    config.norm_valid_dir=f"total_dataset/data_{config.ratio}/norm/valid"
    config.thres_train_dir=f"total_dataset/data_{config.ratio}/thres/train"
    config.thres_valid_dir=f"total_dataset/data_{config.ratio}/thres/valid"
    config.ext_train_dir=f"total_dataset/data_{config.ratio}/ext/train"
    config.ext_valid_dir=f"total_dataset/data_{config.ratio}/ext/valid"
    config.nonext_train_dir=f"total_dataset/data_{config.ratio}/nonext/train"
    config.nonext_valid_dir=f"total_dataset/data_{config.ratio}/nonext/valid"
    config.mean_path=f"total_dataset/data_{config.ratio}/train_mean.json"
    config.std_path=f"total_dataset/data_{config.ratio}/train_std.json"
    config.threshold_path=f"total_dataset/data_{config.ratio}/train_threshold.json"
    return config

def parse_version_jupyter(config):
    from argument_version import add_arguments
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    opt = parser.parse_args('')
    opt = update_config(opt)
    
    opt.no = config['no']
    opt.ratio = config['ratio']
    opt.device = config['device']
    opt.model = config['model']
    opt.method = config['method']
    opt.loss = config['loss']
    opt.batch_size = config['batch_size']
    opt.total_epoch = config['total_epoch']
    opt.gamma = config['gamma']
    opt.patience = config['patience']
    opt.yes = config['yes']
    opt.use_gamma = config['use_gamma']
    opt.use_threshold = config['use_threshold']
    opt.use_min_threshold = config['use_min_threshold']
    opt.only_pm25 = config['only_pm25']
    opt.no_concat_label = config['no_concat_label']
    opt.input_dim = config['input_dim']
    opt.n_heads = config['n_heads']
    opt.origin_all_dir=f"data/origin/all"
    opt.origin_train_dir=f"data/origin/train"
    opt.origin_valid_dir=f"data/origin/valid"
    opt.norm_train_dir=f"data/norm/train"
    opt.norm_valid_dir=f"data/norm/valid"
    opt.thres_train_dir=f"data/thres/train"
    opt.thres_valid_dir=f"data/thres/valid"
    opt.ext_train_dir=f"data/ext/train"
    opt.ext_valid_dir=f"data/ext/valid"
    opt.nonext_train_dir=f"data/nonext/train"
    opt.nonext_valid_dir=f"data/nonext/valid"
    opt.mean_path=f"data/train_mean.json"
    opt.std_path=f"data/train_std.json"
    opt.threshold_path=f"data/train_threshold.json"
    return opt


def update_config(config):
    if not config.no_concat_label:
        config.input_dim += 1
    return config

def save_config(config):
    no = config.no
    method = config.method
    _config = vars(config)
    with open(os.path.join(config.cfg_dir, f"{no}.json"), "w") as fp:
        json.dump(_config, fp, ensure_ascii=False, indent=4)

#################################
########## matrix ###############
#################################
def get_score(y_true, y_pred):
    precision = precision_score  (y_true, y_pred, zero_division=0)
    recall    = recall_score     (y_true, y_pred, zero_division=0)
    f1        = f1_score         (y_true, y_pred, zero_division=0)
    macro     = f1_score         (y_true, y_pred, zero_division=0, average='macro')
    micro     = f1_score         (y_true, y_pred, zero_division=0, average='micro')
    weighted  = f1_score         (y_true, y_pred, zero_division=0, average='weighted')
    mcc       = matthews_corrcoef(y_true, y_pred)
    return precision, recall, f1, macro, micro, weighted, mcc

def get_matrix(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn, fp, fn, tp

#################################
########## model ################
#################################
def get_merged_model(opt, sitename):
    assert opt.nor_load_model != None, f"Merged method should determine the load model"
    assert opt.ext_load_model != None, f"Merged method should determine the load model"
#     nor_load_path = os.path.join(opt.cpt_dir, str(opt.nor_load_model), f"{sitename}.cpt")
    nor_load_path = os.path.join(opt.cpt_dir, str(opt.nor_load_model), f"{sitename}.cpt")
    ext_load_path = os.path.join(opt.cpt_dir, str(opt.ext_load_model), f"{sitename}.cpt")
    nor_model = load_model(nor_load_path, opt)
    ext_model = load_model(ext_load_path, opt)
    for p in nor_model.parameters():
        p.requires_grad = False
    for p in ext_model.parameters():
        p.requires_grad = False
    model = Merged_Model(opt, nor_model, ext_model)
    return model

def load_model(path, opt):
    checkpoint = torch.load(path)
    model = get_model(opt)
    model.load_state_dict(checkpoint)
    return model

def get_model(opt):
    name = opt.model.lower()
    if name == "dnn":
        model = DNN(opt)
    elif name == "rnn":
        model = RNN(opt)
    elif name == "seq":
        model = Seq2Seq(opt)
    elif name == "exf":
        model = Fudan(opt)
    elif name == "evl":
        model = Transformer(opt)
    return model

def get_loss(opt):
    if opt.loss == "bce":
        loss_fn = nn.BCELoss().cuda()
    elif opt.loss == "mse":
        loss_fn = nn.MSELoss().cuda()     
    elif opt.loss == "mae":
        loss_fn = nn.L1Loss().cuda()
    elif opt.loss == "exf":
        loss_fn = ExF(alpha=opt.ratio, gamma=opt.gamma).cuda()
    elif opt.loss == "evl":
        loss_fn = EVL(alpha=opt.ratio, gamma=opt.gamma).cuda()
    return loss_fn

def get_trainer(opt):
    if opt.method == "exf":
        trainer = ExF_trainer
    elif opt.method == "class":
        trainer = class_trainer
    elif opt.method == "evl":
        trainer = EVL_trainer
    elif opt.method == "merged":
        trainer = merged_trainer
    else:
        raise ValueError(f"--method does not support {opt.method}")
    return trainer

#################################
########## dataset ##############
#################################
def get_dataset(opt, sitename, isTrain):
    from dataset import PMDataset, PMFudanDataset
    if opt.method == "ExF":
        return PMFudanDataset(sitename=sitename, opt=opt, isTrain=isTrain)
    else:
        return PMDataset(sitename=sitename, opt=opt, isTrain=isTrain)

def read_file(sitename, opt, mode, isTrain):
    if mode == 0:
        read_path = os.path.join(opt.origin_train_dir, f"{sitename}.npy") if isTrain else os.path.join(opt.origin_valid_dir, f"{sitename}.npy")
    elif mode == 1:
        read_path = os.path.join(opt.thres_train_dir, f"{sitename}.npy") if isTrain else os.path.join(opt.thres_valid_dir, f"{sitename}.npy")
    if os.path.exists(read_path):
        data = np.load(read_path)
    else:
        raise ValueError(f"path {read_path} doesn't exist")
    return data

def get_mask(opt, data, thres_data):
    mask = np.zeros((data.shape[0], 1))
    if opt.use_threshold:
        if opt.use_min_threshold:
            index = np.argwhere(thres_data[:, 7] >= opt.threshold)
            thres_data[index, 7] = opt.threshold
        mask[data[:, 7]>=thres_data[:, 7]] = 1
    if opt.use_delta:
        dif_data = abs(data[1:, 7] - data[:-1, 7]) if opt.use_abs_delta else data[1:, 7] - data[:-1, 7]
        index = np.argwhere(dif_data>=opt.delta)[:, 0] + 1
        mask[index] = 1
    return mask

def get_split_dataset(opt, data, mask):
    """
        data: [data len, input_dim] 
        mask: [data len, 1]
        mode: 'norm', 'ext'
    """
    size = data.shape[0]
    shift = opt.memory_size + opt.source_size + opt.target_size
    _data = []
    _mask = []
    if opt.split_mode == "norm":
        for i in range(size - shift) :
            data_patch = data[i: i + shift]
            mask_patch = mask[i: i + shift]
            if np.sum(mask_patch[shift - opt.target_size:]) < 1:
                _data.append(data_patch)
                _mask.append(mask_patch)
    elif opt.split_mode == "ext":
        for i in range(size - shift) :
            data_patch = data[i: i + shift]
            mask_patch = mask[i: i + shift]
            if np.sum(mask_patch[shift - opt.target_size:]) > 0:
                _data.append(data_patch)
                _mask.append(mask_patch)

    return np.array(_data), np.array(_mask)
