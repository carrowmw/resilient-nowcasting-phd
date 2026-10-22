import sys, numpy as np, pandas as pd, glob
sys.path.insert(0,'src')
from quality.stats import diebold_mariano
from statsmodels.stats.multitest import multipletests
R='artifacts/spatial_full/runs'
def load(model,graph):
    fs=sorted(glob.glob(f'{R}/{model}_{graph}_s*/predictions.parquet'))
    parts=[]
    for f in fs:
        d=pd.read_parquet(f,columns=['node_id','horizon','target_dt','y_true_raw','y_pred_raw','target_observed'])
        parts.append(d[d.target_observed].drop(columns='target_observed'))
    d=pd.concat(parts)
    g=d.groupby(['node_id','horizon','target_dt'],sort=False).agg(y=('y_true_raw','first'),p=('y_pred_raw','mean'),k=('y_pred_raw','size')).reset_index()
    assert g.k.nunique()==1, (model, g.k.unique())
    return g, len(fs)
def nw_bartlett(d, L):
    n=len(d); dc=d-d.mean(); v=np.mean(dc*dc)
    for l in range(1,L+1):
        v+=2*(1-l/(L+1))*np.mean(dc[l:]*dc[:-l])
    return d.mean()/np.sqrt(v/n)
champ,ns=load('gat_lstm','gabriel'); print('champ seeds',ns)
comps=[('lstm','identity'),('transformer','identity'),('gat_transformer','identity'),('gat_transformer','gabriel'),('snaive_weekly','identity'),('persistence','identity'),('hist_avg','identity')]
rows=[];per=[]
from scipy import stats as sps
for m,g in comps:
    b,nb=load(m,g)
    mg=champ.merge(b,on=['node_id','horizon','target_dt'],suffixes=('_a','_b'))
    assert len(mg)==len(champ)==len(b),(m,len(mg),len(champ),len(b))
    assert np.allclose(mg.y_a,mg.y_b)
    for h in (1,2,4):
        sub=mg[mg.horizon==h]
        recs=[]
        for node,grp in sub.groupby('node_id'):
            grp=grp.sort_values('target_dt')
            ea=(grp.p_a-grp.y_a).to_numpy(); eb=(grp.p_b-grp.y_a).to_numpy()
            r=diebold_mariano(ea,eb,horizon=h,loss='absolute')
            d=np.abs(ea)-np.abs(eb)
            z96=nw_bartlett(d,96); p96=2*sps.norm.sf(abs(z96))
            recs.append(dict(comparison=f'gat_lstm|gabriel vs {m}|{g}',horizon=h,node_id=node,n=r['n'],dm_stat=r['dm_stat'],p_value=r['p_value'],mean_loss_diff=r['mean_loss_diff'],dm_nw96=z96,p_nw96=p96,
                             mae_a=np.abs(ea).mean(),mae_b=np.abs(eb).mean()))
        df=pd.DataFrame(recs)
        rej,pf,_,_=multipletests(df.p_value,alpha=0.05,method='fdr_bh'); df['p_bh']=pf; df['sig_bh']=rej
        rej96,pf96,_,_=multipletests(df.p_nw96,alpha=0.05,method='fdr_bh'); df['sig_bh_nw96']=rej96
        per.append(df)
        better=df.mean_loss_diff<0
        rows.append(dict(comparison=f'gat_lstm|gabriel vs {m}|{g}',horizon=h,seeds_focal=ns,seeds_baseline=nb,n_nodes=len(df),
            pct_nodes_better=100*better.mean(),pct_sig_bh=100*df.sig_bh.mean(),pct_sig_bh_champion_better=100*(df.sig_bh&better).mean(),
            pct_sig_bh_baseline_better=100*(df.sig_bh&~better).mean(),
            median_dm_stat=df.dm_stat.median(),mean_loss_diff_counts=df.mean_loss_diff.mean(),median_loss_diff_counts=df.mean_loss_diff.median(),
            pct_sig_bh_champion_better_nw96=100*(df.sig_bh_nw96&better).mean(),median_dm_nw96=df.dm_nw96.median()))
s=pd.DataFrame(rows)
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
print(s.round(3).to_string())
s.to_csv('docs/thesis/ch6_new_figures/dm_champion_seedavg.csv',index=False)
pd.concat(per).to_csv('/tmp/dm_champion_seedavg_pernode.csv',index=False)
