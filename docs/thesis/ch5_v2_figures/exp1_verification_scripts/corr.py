from load import *
import sys; sys.path.insert(0,'/workspace/src')
from quality.stats import partial_spearman, segmented_threshold_fit
import scipy.stats as sps
feats=['completeness_active','avg_seq_len_hours','diurnal_capture_ratio','mean_count_raw']
for name,f in [('MIXED(v1 err+v2 quality; plot scripts)',v1),('V2 (canonical runs)',v2)]:
  for h in (8,24):
    d=f(h=h)
    y=d.cv_rmse_raw.values; vol=d.mean_count_raw.values
    out=[]
    for c in feats:
      r,p=sps.spearmanr(d[c],y)
      pr=partial_spearman(d[c].values,y,vol[:,None])[0] if c!='mean_count_raw' else np.nan
      out.append(f'{c}: rho={r:.3f} (p={p:.3g}) partial|vol={pr:.3f}')
    print(name,'h',h,'n',len(d)); print('  '+'\n  '.join(out))
    print('  rho(seq,vol)=%.3f rho(seq,comp)=%.3f'%(sps.spearmanr(d.avg_seq_len_hours,vol)[0],sps.spearmanr(d.avg_seq_len_hours,d.completeness_active)[0]))
    print('  partial seq|comp=%.3f p=%.3g ; comp|seq=%.3f p=%.3g'%(*partial_spearman(d.avg_seq_len_hours.values,y,d[['completeness_active']].values),*partial_spearman(d.completeness_active.values,y,d[['avg_seq_len_hours']].values)))
    if h==8:
      fit=segmented_threshold_fit(d.avg_seq_len_hours.values,y,n_bootstrap=1000,seed=42)
      print('  threshold bp=%.2f CI=[%.2f,%.2f] below=%.2f above=%.2f dAIC=%.2f'%(fit.breakpoint,*fit.breakpoint_ci,fit.slope_below,fit.slope_above,fit.delta_aic))
