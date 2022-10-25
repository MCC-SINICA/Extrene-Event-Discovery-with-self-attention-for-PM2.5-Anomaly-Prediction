import torch
import torch.nn as nn
from tqdm import tqdm
from custom_loss import *


def ExF_trainer(opt, dataloader, model, loss_fn, optimizer=None):
    if optimizer:
        model.train()
    else:
        model.eval()
    mean_pred_loss = 0
    mean_rmse_loss = 0
    trange = tqdm(dataloader)
    mse_fn = nn.MSELoss().cuda()
    for idx, data in enumerate(trange):
        # get data
        x, y_true, ext_true, thres_y, past_window, past_ext = map(lambda z: z.cuda(), data)
        # forward
        y_pred, ext_pred, past_pred = model(x, past_window, past_ext)
        # Calculate loss
        mse_loss = mse_fn (y_pred,    y_true)
        ext_loss = loss_fn(ext_pred,  ext_true)
        his_loss = loss_fn(past_pred, past_ext)
        loss = mse_loss + ext_loss + his_loss
        if optimizer:
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Record loss
        with torch.no_grad():
            rmse_loss  = torch.sqrt(mse_fn(y_pred * thres_y, y_true * thres_y)) 
            mean_rmse_loss += rmse_loss.item()
            mean_pred_loss += ext_loss.item()

    mean_rmse_loss /= len(dataloader)
    mean_pred_loss /= len(dataloader)
    return mean_pred_loss

def class_trainer(opt, dataloader, model, loss_fn, optimizer=None):
    if optimizer:
        model.train()
    else:
        model.eval()
    mean_loss = 0
    trange = tqdm(dataloader)
    for idx, data in enumerate(trange):
        # get data
        x, y_true, ext_true, past_data = map(lambda z: z.cuda(), data)
        # get loss & update
        if opt.model.lower() == "seq":
            ext_pred = model(past_data, x) 
        else:
            # emb, hid, out
            _, _, ext_pred = model(past_data, x)
        
        # Calculate loss
        loss = loss_fn(ext_pred, ext_true)
        if optimizer:
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Record loss
        mean_loss += loss.item()

    mean_loss /= len(dataloader)
    return mean_loss

def EVL_trainer(opt, dataloader, model, loss_fn, optimizer=None):
    if optimizer:
        model.train()
    else:
        model.eval()
    mean_loss = 0
    trange = enumerate(dataloader)
    for idx, data in trange:
        # get data
        x, y_true, ext_true, past_data = map(lambda z: z.cuda(), data)
        # get loss & update
        ext_pred, _, _ = model(past_data, x)
        # Calculate loss
        loss = loss_fn(ext_pred, ext_true)
        if optimizer:
            # Update model
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
            optimizer.step()
        # Record loss
        mean_loss += loss.item()

    mean_loss /= len(dataloader)
    return mean_loss

def merged_trainer(opt, dataloader, model, loss_fn, optimizer=None):
    if optimizer:
        model.train()
    else:
        model.eval()
    mean_loss = 0
    trange = enumerate(dataloader)
    for idx, data in trange:
        # get data
        x, y_true, ext_true, past_data = map(lambda z: z.cuda(), data)
        # get loss & update
        ext_pred = model(past_data, x)
        # Calculate loss
        loss = loss_fn(ext_pred, ext_true)
        if optimizer:
            # Update model
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
            optimizer.step()
        # Record loss
        mean_loss += loss.item()

    mean_loss /= len(dataloader)
    return mean_loss
