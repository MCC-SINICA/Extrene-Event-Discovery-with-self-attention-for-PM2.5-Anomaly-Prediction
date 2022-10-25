from utils import *
opt = parse(data_version=1)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.device)
from constants import *
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import *
import csv, json
import pandas as pd
import pygsheets

# Test
with open(f"{opt.cfg_dir}/{opt.no}.json", "r") as fp: # opt.cfg_dir : configs/
    opt = json.load(fp)
opt = Namespace(**opt)
same_seeds(opt.seed)

no = opt.no

same_seeds(opt.seed)
cpt_dir = os.path.join(opt.cpt_dir, str(no))
rst_dir = os.path.join(opt.rst_dir, str(no))
if not os.path.exists(rst_dir):
    os.makedirs(rst_dir, 0o777)

results = []
mean_precision, mean_recall, mean_f1, mean_mcc = 0,0,0,0
for sitename in SITENAMES:
    if opt.skip_site and sitename not in SAMPLE_SITES:
        continue
    # Dataset
    dataset = get_dataset(opt=opt, sitename=sitename, isTrain=False)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)
    # Model
    if opt.method == "merged":
        model = get_merged_model(opt, sitename)
    else:
        model = get_model(opt)
    # Load checkpoint
    model.load_state_dict(torch.load(os.path.join(cpt_dir, f"{sitename}.cpt")))
    # For device
    model.cuda()
    # Freeze model
    model.eval()
    # Parameters
    st_time = datetime.now()
    pred_list = None
    true_list = None
    trange = tqdm(dataloader)
    for data in trange:
        trange.set_postfix_str(sitename)
        # get data
        if opt.method == "ExF":
            x, y_true, ext_true, thres_y, past_data, past_ext = map(lambda z: z.cuda(), data)
        else:
            x, y_true, ext_true, past_data = map(lambda z: z.cuda(), data)
            
        # get prediction
        if opt.method == "merged":
            ext_pred = model(past_data, x)
        else:
            if opt.model == "ExF":
                _, ext_pred, _ = model(x, past_data, past_ext)
            elif opt.model == "EVL":
                ext_pred, _, _ = model(past_data, x)
            elif opt.model == "seq":
                ext_pred = model(past_data, x)
            else:
                _, _, ext_pred = model(past_data, x)
        # Recover predict
        ext_pred[ext_pred>=0.5] = 1
        ext_pred[ext_pred<0.5]  = 0
        ext_pred = ext_pred.detach().cpu().numpy()
        ext_true = ext_true.detach().cpu().numpy()
        # Append result
        if pred_list is None:
            pred_list = ext_pred
            true_list = ext_true
        else:
            pred_list = np.concatenate((pred_list, ext_pred), axis=0)
            true_list = np.concatenate((true_list, ext_true), axis=0)
    # Save results
    np.save(f"{rst_dir}/{sitename}.npy", pred_list)
    j = -1
    precision, recall, f1, macro, micro, weighted, mcc = get_score(true_list[:, j], pred_list[:, j])
    tn, fp, fn, tp = get_matrix(true_list[:, j], pred_list[:, j])
    mean_precision += precision; mean_recall += recall; mean_f1 += f1; mean_mcc += mcc;
    results.append({
        'sitename': sitename,
        'precision': f"{precision:.4f}",
        'recall'   : f"{recall   :.4f}",
        'f1'       : f"{f1       :.4f}",
        'mcc'      : f"{mcc      :.4f}",
        'tn'       : f"{tn       :.4f}",
        'fp'       : f"{fp       :.4f}",
        'fn'       : f"{fn       :.4f}",
        'tp'       : f"{tp       :.4f}",
    })
results.insert(0, {
    'sitename': 'average',
    'precision': f"{mean_precision /len(SITENAMES):.4f}",
    'recall'   : f"{mean_recall    /len(SITENAMES):.4f}",
    'f1'       : f"{mean_f1        /len(SITENAMES):.4f}",
    'mcc'      : f"{mean_mcc       /len(SITENAMES):.4f}",
})
df = pd.DataFrame(results, columns = ['sitename', 'precision', 'recall', 'f1', 'mcc', 'tp', 'tn', 'fp','fn']) 
df.to_csv(f"{rst_dir}/{no}.csv", index=False, encoding='utf_8_sig')
print(f"Finish testing no: {no}, cost time: {datetime.now()-st_time}")