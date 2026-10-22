import pandas as pd, numpy as np, json
W='/workspace/artifacts/'
man=json.load(open(W+'quality_full_v2/sensor_manifest.json'))
viable=man['viable_sensors']
q=pd.read_parquet(W+'quality_full_v2/quality_table_none.parquet')
res=pd.read_parquet(W+'quality_full/analysis/results.parquet')
def v1(model='lstm_mv',h=8,interp='none'):
    s=res[(res.model==model)&(res.horizon==h)&(res.interpolation==interp)&(res.ablation=='')&(res.sensor_id.isin(viable))].copy()
    s=s.drop(columns=[c for c in q.columns if c in s.columns])
    return s.merge(q.reset_index(),on='sensor_id',how='left').set_index('sensor_id')
def v2(model='lstm_mv',h=8,interp='none'):
    p=pd.read_parquet(W+f'quality_full_v2/runs/{model}_h{h}_{interp}/predictions.parquet')
    p=p[p.split=='test']
    if 'target_interpolated' in p and interp!='none': pass
    g=p.groupby('sensor_id').apply(lambda d: pd.Series({'rmse_raw':np.sqrt(((d.y_true_raw-d.y_pred_raw)**2).mean()),'mean_count_raw':d.y_true_raw.mean(),'n':len(d)}))
    g['cv_rmse_raw']=100*g.rmse_raw/g.mean_count_raw
    return g.join(q)
