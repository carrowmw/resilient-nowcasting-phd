from load import *
import json, scipy.stats as sps
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
places=json.load(open('/workspace/data/external/places_cache.json'))
F=["avg_seq_len_hours","completeness_active","diurnal_capture_ratio","mean_count_raw","total_poi_count","transit_count","dist_to_city_centre"]
for lab,f in [('mixed (thesis figs)',lambda h: v1(h=h)),('v2',lambda h: v2(h=h))]:
  for h in (8,24):
    d=f(h).copy(); d['loc']=[i.split('__')[0] for i in d.index]
    d=d[d['loc'].isin(places)]
    for c in ['total_poi_count','transit_count','dist_to_city_centre']: d[c]=d['loc'].map(lambda l: places[l].get(c,0))
    d=d.dropna(subset=F+['cv_rmse_raw'])
    X=StandardScaler().fit_transform(d[F])
    sil={k:round(silhouette_score(X,KMeans(n_clusters=k,random_state=42,n_init=10).fit(X).labels_),3) for k in range(2,7)}
    lab_=KMeans(n_clusters=2,random_state=42,n_init=10).fit(X).labels_; d['c']=lab_
    core=d.groupby('c').dist_to_city_centre.mean().idxmin()
    a=d[d.c==core].cv_rmse_raw; b=d[d.c!=core].cv_rmse_raw
    U,p=sps.mannwhitneyu(a,b,alternative='two-sided')
    # location-level (avoid paired directions)
    print(f'{lab} h{h} n={len(d)} core n={len(a)} sub n={len(b)} median core {a.median():.1f} sub {b.median():.1f} MWU U={U:.0f} p={p:.4f} | silhouette {sil}')
    print('   core locations',d[d.c==core]['loc'].nunique(),'sub locations',d[d.c!=core]['loc'].nunique(), 'locations split across clusters', (d.groupby('loc').c.nunique()>1).sum())
    if h==8 and lab.startswith('mixed'):
        import numpy as np
        print('   rho(dist,cv)=%.3f'%sps.spearmanr(d.dist_to_city_centre,d.cv_rmse_raw)[0])
