from load import *
from sklearn.ensemble import RandomForestRegressor
import warnings; warnings.filterwarnings('ignore')
def v1own(h):
    s=res[(res.model=='lstm_mv')&(res.horizon==h)&(res.interpolation=='none')&(res.ablation=='')]
    return s[s.sensor_id.isin(viable)].set_index('sensor_id')
FS={'A comp,seq,diur,vol':['completeness_active','avg_seq_len_hours','diurnal_capture_ratio','mean_count_raw'],
    'B comp,seq,std,diur':['completeness_active','avg_seq_len_hours','std_seq_len_hours','diurnal_capture_ratio'],
    'C comp,seq,diur':['completeness_active','avg_seq_len_hours','diurnal_capture_ratio'],
    'D comp,seq,seqcount,diur':['completeness_active','avg_seq_len_hours','seq_count','diurnal_capture_ratio']}
for nm,f in {'v1own':v1own,'mixed':lambda h:v1(h=h),'v2':lambda h:v2(h=h)}.items():
  for h in (8,24):
    d=f(h)
    for k,fs in FS.items():
      for tgt in ['cv_rmse_raw','rmse_raw']:
        rf=RandomForestRegressor(n_estimators=300,random_state=42,oob_score=True).fit(d[fs].values,d[tgt].values)
        print(nm,h,k,tgt,round(rf.oob_score_,3))
