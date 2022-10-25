import numpy as np
import os, shutil
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style({u'font.sans-serif':['simhei']})
plt.rcParams[u'font.sans-serif'] = ['simhei']
plt.rcParams['axes.unicode_minus'] = False
import pandas as pd
import scipy
import math
from scipy.stats import genextreme as gev, gumbel_r, norm, gompertz
from scipy.special import gamma, factorial
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef

from constants import *
from utils import *

def gussion(x, position, width, height):
    return  height * math.sqrt(2*math.pi) * width * scipy.stats.norm.pdf(x, position, width)

def extreme(x, position, width, height, c=-0.1):
    return  height * math.sqrt(2*math.pi) * width * gev.pdf(x, c, position, width)

def kld(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


opt = parse()
with open(f"{opt.cfg_dir}/{opt.no}.json", "r") as fp:
    opt = json.load(fp)
opt = Namespace(**opt)
same_seeds(opt.seed)
origin_all_path = opt.origin_all_dir
st_time = datetime.now()

results = []
for idx, sitename in enumerate(SITENAMES):
    print(sitename)
    read_path = os.path.join(origin_all_path, f"{sitename}.npy")
    data = np.load(read_path)[:, 7].astype(np.int)
    data[data<0] = 0
    ratio = 1
    max_bin = int(np.max(data)) + 1
    x = np.linspace(0, max_bin-1, max_bin-1)
    bins = [i * ratio for i in range(max_bin)]
    hist, _ = np.histogram(data, bins=bins, density=True)

    gev_fit = gev.fit(data)
    gev_dist = gev.pdf(x, gev_fit[0], gev_fit[1], gev_fit[2])
    gev_kl = kld(hist, gev_dist)
    
    norm_fit = norm.fit(data)
    norm_dist = norm.pdf(x, norm_fit[0], norm_fit[1])
    norm_kl = kld(hist, norm_dist)
    
    results.append({
        'sitename' : sitename,
        'gev_shape': f"{gev_fit[0]:.4f}",
        'gev_loc'  : f"{gev_fit[1]:.4f}",
        'gev_scl'  : f"{gev_fit[2]:.4f}",
        'norm_loc' : f"{norm_fit[0]:.4f}",
        'norm_scl' : f"{norm_fit[1]:.4f}",
        'gev_kl'   : f"{gev_kl:.4f}",
        'norm_kl'  : f"{norm_kl:.4f}",
    })
    
     
df = pd.DataFrame(results) 
df.to_csv(f"analysis/long_tail_analysis.csv", index=False, encoding='utf_8_sig')
print(f"Finish cost time: {datetime.now()-st_time}")
