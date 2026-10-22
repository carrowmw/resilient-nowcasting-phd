from load import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_score, RepeatedKFold
import warnings; warnings.filterwarnings('ignore')
QP=['completeness_active','avg_seq_len_hours','std_seq_len_hours','seq_count']
def v1own(h):
    s=res[(res.model=='lstm_mv')&(res.horizon==h)&(res.interpolation=='none')&(res.ablation=='')]
    return s[s.sensor_id.isin(viable)].set_index('sensor_id')
srcs={'v1own':v1own,'mixed':lambda h: v1(h=h),'v2':lambda h: v2(h=h)}
for nm,f in srcs.items():
  for h in (8,24):
    d=f(h); g=np.array([i.split('__')[0] for i in d.index])
    for tgt in ['cv_rmse_raw','rmse_raw']:
      for feats in [QP, QP+['mean_count_raw']]:
        X=d[feats].values; y=d[tgt].values
        rf=RandomForestRegressor(n_estimators=300,random_state=42,oob_score=True).fit(X,y)
        rk=cross_val_score(RandomForestRegressor(n_estimators=300,random_state=42),X,y,cv=RepeatedKFold(n_splits=5,n_repeats=10,random_state=42),scoring='r2')
        gk=cross_val_score(RandomForestRegressor(n_estimators=300,random_state=42),X,y,cv=GroupKFold(n_splits=5),groups=g,scoring='r2')
        # pooled grouped R2
        from sklearn.model_selection import cross_val_predict
        pp=cross_val_predict(RandomForestRegressor(n_estimators=300,random_state=42),X,y,cv=GroupKFold(n_splits=5),groups=g)
        r2p=1-((y-pp)**2).sum()/((y-y.mean())**2).sum()
        print(f"{nm} h{h} {tgt:12s} {'+vol' if len(feats)>4 else '4q  '} n={len(d)} OOB={rf.oob_score_:.3f} RKF5x10 mean={rk.mean():.3f} med={np.median(rk):.3f} GroupKF5 foldmean={gk.mean():.3f} pooled={r2p:.3f}")
