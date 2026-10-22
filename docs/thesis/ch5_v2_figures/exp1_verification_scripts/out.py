import pandas as pd, numpy as np
W='/workspace/artifacts/quality_full_v2/'
p=pd.read_parquet(W+'runs/lstm_mv_h8_none/predictions.parquet')
q=pd.read_parquet(W+'quality_table_none.parquet')
def met(d): r=np.sqrt(((d.y_pred_raw-d.y_true_raw)**2).mean()); return r, 100*r/d.y_true_raw.mean()
u=p[p.sensor_id=='NclUrbanScienceBuildingSquareMid__west_to_east'].copy()
u['err']=u.y_pred_raw-u.y_true_raw
print('US n',len(u),'mean %.2f median %.1f max %.0f'%(u.y_true_raw.mean(),u.y_true_raw.median(),u.y_true_raw.max()), 'RMSE/CV',met(u))
print(q.loc['NclUrbanScienceBuildingSquareMid__west_to_east',['completeness_active','avg_seq_len_hours','seq_count','max_seq_len_hours','first_obs','last_obs','test_windows_h8']].to_dict())
print(u.reindex(u.err.abs().sort_values(ascending=False).index)[['origin_dt','target_dt','y_true_raw','y_pred_raw']].head(10).to_string())
top=u.err.abs().sort_values(ascending=False)
for k in (1,3,5,10):
    v=u.drop(top.index[:k]); print('drop top',k,'RMSE %.2f CV %.2f'%met(v))
print('share of SSE from top 5: %.3f'%((top.iloc[:5]**2).sum()/(top**2).sum()))
print('target date range',u.target_dt.min(),u.target_dt.max())
# sibling direction
u2=p[p.sensor_id=='NclUrbanScienceBuildingSquareMid__east_to_west']; print('east_to_west RMSE/CV',met(u2),'max',u2.y_true_raw.max())
# raw data duplicates check for the max reading
raw=pd.read_parquet('/workspace/data/processed/master_dataset_15T.parquet',columns=['dt','value','location','veh_class','dir'],filters=[('location','in',['NclUrbanScienceBuildingSquareMid','NclHeberSt']),('veh_class','==','person')])
r=raw[(raw.location=='NclUrbanScienceBuildingSquareMid')&(raw.veh_class=='person')]
print(r.dir.unique()); rw=r[r.dir=='west_to_east']
print('raw w2e n',len(rw),'dup dt',rw.dt.duplicated().sum(),'top values',rw.nlargest(8,'value')[['dt','value']].to_string())
# Heber
h=p[p.sensor_id=='NclHeberSt__east_to_west'].copy()
print('Heber RMSE/CV',met(h)); i=h.y_true_raw.idxmax(); print(h.loc[i,['origin_dt','target_dt','y_true_raw','y_pred_raw']].to_dict())
h2=h.copy(); h2.loc[i,'y_true_raw']=h2.loc[i,'y_true_raw']/2; print('dedup (halve) RMSE %.2f CV %.2f'%met(h2))
print('omit RMSE %.2f CV %.2f'%met(h.drop(i)))
print('non-surge <=260 RMSE %.2f CV %.2f'%met(h[h.y_true_raw<=260]), 'n surge',(h.y_true_raw>260).sum())
rh=raw[(raw.location=='NclHeberSt')&(raw.veh_class=='person')&(raw.dir=='east_to_west')]
print('raw Heber dup dt',rh.dt.duplicated().sum(), rh.nlargest(5,'value')[['dt','value']].to_string())
