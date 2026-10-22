import pandas as pd, numpy as np, scipy.stats as sps
W='/workspace/artifacts/quality_full_v2/runs/'
pn=pd.read_parquet(W+'lstm_mv_h8_none/predictions.parquet'); pk=pd.read_parquet(W+'lstm_mv_h8_knn/predictions.parquet')
print('none rows',len(pn),'knn rows',len(pk),'ratio %.3f'%(len(pk)/len(pn)), 'knn interp targets',pk.target_interpolated.sum(), 'knn real targets', (~pk.target_interpolated).sum())
m=pn[['sensor_id','target_dt','y_true_raw','y_pred_raw']].merge(pk[['sensor_id','target_dt','y_true_raw','y_pred_raw','target_interpolated']],on=['sensor_id','target_dt'],suffixes=('_none','_knn'))
print('matched',len(m),'sensors',m.sensor_id.nunique(), 'ytrue equal', np.allclose(m.y_true_raw_none,m.y_true_raw_knn), 'knn target interp in matched',m.target_interpolated.sum())
missing=sorted(set(pn.sensor_id)-set(m.sensor_id)); print('missing from matched',missing)
for s in missing:
    a=pn[pn.sensor_id==s]; b=pk[pk.sensor_id==s]
    print(s,'none test n',len(a),a.target_dt.min(),a.target_dt.max(),'| knn test n',len(b),b.target_dt.min(),b.target_dt.max())
print('date span',m.target_dt.min(),m.target_dt.max())
rn=np.sqrt(((m.y_pred_raw_none-m.y_true_raw_none)**2).mean()); rk=np.sqrt(((m.y_pred_raw_knn-m.y_true_raw_knn)**2).mean()); mu=m.y_true_raw_none.mean()
print('pooled RMSE none %.2f knn %.2f ; pooled CV none %.2f knn %.2f diff %.2f pp; rel RMSE change %.1f%%'%(rn,rk,100*rn/mu,100*rk/mu,100*(rk-rn)/mu,100*(rk-rn)/rn))
def sm(g):
    r_n=np.sqrt(((g.y_pred_raw_none-g.y_true_raw_none)**2).mean()); r_k=np.sqrt(((g.y_pred_raw_knn-g.y_true_raw_knn)**2).mean()); mu=g.y_true_raw_none.mean()
    return pd.Series({'n':len(g),'cv_n':100*r_n/mu,'cv_k':100*r_k/mu})
s=m.groupby('sensor_id').apply(sm); d=s.cv_k-s.cv_n
print('sensor mean CV none %.2f knn %.2f; median %.2f %.2f; diff mean %.2f sd %.2f; median diff %.2f; none better on %d/%d'%(s.cv_n.mean(),s.cv_k.mean(),s.cv_n.median(),s.cv_k.median(),d.mean(),d.std(),d.median(),(d>0).sum(),len(d)))
print('paired t',sps.ttest_rel(s.cv_k,s.cv_n),'wilcoxon',sps.wilcoxon(s.cv_k,s.cv_n))
print('min n matched per sensor',s.n.min(),'median',s.n.median())
h=m.target_dt.dt.hour; print('frac 13:00-22:59 UTC %.3f ; 13-21 incl %.3f; 00-08 %.4f'%(((h>=13)&(h<=22)).mean(),((h>=13)&(h<=21)).mean(),(h<8).mean()))
print('tz',m.target_dt.dt.tz)
print(h.value_counts(normalize=True).sort_index().round(3).to_dict())
# hourly
m['h']=h
hh=m.groupby('h').apply(lambda g: pd.Series({'n':len(g),'rn':np.sqrt(((g.y_pred_raw_none-g.y_true_raw_none)**2).mean()),'rk':np.sqrt(((g.y_pred_raw_knn-g.y_true_raw_knn)**2).mean())}))
print(hh.round(2).to_string())
