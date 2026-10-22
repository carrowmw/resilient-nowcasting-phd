from load import *
import sys; sys.path.insert(0,'/workspace/src')
from quality.stats import partial_spearman
import scipy.stats as sps
feats=['completeness_active','avg_seq_len_hours','diurnal_capture_ratio','mean_count_raw']
for h in (8,24):
  for lab,filt in [('v1-own 110',True),('v1-own 114',False)]:
    s=res[(res.model=='lstm_mv')&(res.horizon==h)&(res.interpolation=='none')&(res.ablation=='')]
    if filt: s=s[s.sensor_id.isin(viable)]
    y=s.cv_rmse_raw.values; vol=s.mean_count_raw.values
    print(lab,h,len(s),' '.join(f"{c[:6]}={sps.spearmanr(s[c],y)[0]:.3f}/{(partial_spearman(s[c].values,y,vol[:,None])[0] if c!='mean_count_raw' else np.nan):.3f}" for c in feats), 'seqvol=%.3f'%sps.spearmanr(s.avg_seq_len_hours,vol)[0])
