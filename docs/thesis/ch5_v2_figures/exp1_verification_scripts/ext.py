import pandas as pd, numpy as np
W='/workspace/artifacts/quality_full_v2/runs/'
for arm in ['none','knn']:
  p=pd.read_parquet(W+f'lstm_mv_h8_{arm}/predictions.parquet'); p=p[p.split=='test']
  rows=[]; band_all=[]
  for s,g in p.groupby('sensor_id'):
    y=g.y_true_raw.values; yp=g.y_pred_raw.values
    mu=y.mean(); sd=y.std(ddof=1)
    if np.isnan(sd) or sd<1: continue
    ext=y>mu+3*sd; nor=(y<=mu+sd)&(y>=mu-sd); mid=(~ext)&(~nor)
    if not ext.any() or not nor.any(): continue
    def m(mask):
        r=np.sqrt(np.mean((yp[mask]-y[mask])**2)); return r,100*r/y[mask].mean(),100*np.mean(np.abs(yp[mask]-y[mask])/np.maximum(np.abs(y[mask]),1)),y[mask].mean(),np.median(y[mask])
    rn=m(nor); re=m(ext)
    rows.append(dict(sensor=s,rmse_n=rn[0],cv_n=rn[1],mape_n=rn[2],mean_n=rn[3],rmse_e=re[0],cv_e=re[1],mape_e=re[2],mean_e=re[3],med_e=re[4],frac_ext=ext.mean(),frac_mid=mid.mean(),frac_nor=nor.mean(),n_ext=ext.sum(),maxres=np.max(np.abs(yp[ext]-y[ext]))))
    band_all.append(y[ext])
  d=pd.DataFrame(rows)
  print(arm,'sensors',len(d))
  print(' median RMSE normal %.2f extreme %.2f ratio %.2f'%(d.rmse_n.median(),d.rmse_e.median(),d.rmse_e.median()/d.rmse_n.median()))
  print(' median CV normal %.1f extreme %.1f ; median MAPE normal %.1f extreme %.1f'%(d.cv_n.median(),d.cv_e.median(),d.mape_n.median(),d.mape_e.median()))
  print(' per-sensor mean y in extreme band: median %.1f, IQR %.1f-%.1f, min %.1f max %.1f; sensors with extreme-band mean>=100: %d'%(d.mean_e.median(),d.mean_e.quantile(.25),d.mean_e.quantile(.75),d.mean_e.min(),d.mean_e.max(),(d.mean_e>=100).sum()))
  ae=np.concatenate(band_all); print(' pooled extreme-band y_true median %.0f, mean %.1f, frac>=100 %.2f, frac>=1000 %.4f, n %d'%(np.median(ae),ae.mean(),(ae>=100).mean(),(ae>=1000).mean(),len(ae)))
  print(' median frac of steps: normal %.3f, 1-3sigma band(excluded) %.3f, extreme %.3f; max residual in extreme %.0f'%(d.frac_nor.median(),d.frac_mid.median(),d.frac_ext.median(),d.maxres.max()))
  print(' median per-sensor mean y normal band %.1f'%d.mean_n.median())
