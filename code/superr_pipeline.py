"""SupeRR Selector S2 - full pipeline (run scripts in order; each saves pickles for the next).
Stages: s1 match states + win-prob model + leverage | s2 pressure index | s3 context baselines |
s7 run-value metric | s8 empirical-Bayes rating, consistency, bootstrap, sensitivity | s9 export."""

# ======== s1_model.py ========
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
import os; os.makedirs('results', exist_ok=True)
U='data/raw/'
b=pd.read_csv(U+'ball_by_ball.csv',low_memory=False)
m=pd.read_csv(U+'matches.csv'); c=pd.read_csv(U+'competitions.csv')
b=b.merge(m[['match_id','comp_id','match_date','win_team_id','result','first_inn_revised_overs','second_inn_revised_overs','revised_target','venue_id']],on='match_id').merge(c[['comp_id','comp_name','match_format','season']],on='comp_id')
b['is_ipl']=b.comp_name.str.startswith('IPL').astype(int)
b=b[((b.is_ipl==1)|(b.match_format=='T20I'))&(b.innings<=2)].copy()
# drop rain-affected / no-result matches
bad=set(b.loc[b.revised_target.notna()|b.first_inn_revised_overs.notna()|b.second_inn_revised_overs.notna(),'match_id'])
nores=set(m.loc[~m.result.fillna('').str.contains('Won'),'match_id'])
b=b[~b.match_id.isin(bad|nores)].copy()
b=b[b.win_team_id.isin(b.batting_team_id.unique())|True]
b=b.sort_values(['match_id','innings','innings_ball_num']).reset_index(drop=True)
et=b.extras_type.fillna('')
b['legal']=(~et.isin(['wide','Overstep','other'])).astype(int)
b['xtra']=np.where(et.isin(['wide','Overstep','other','legbyes','byes']),1,0)
b['tot']=b.runs+b.xtra
b['wkt']=b.is_wicket.astype(int)
g=b.groupby(['match_id','innings'])
b['legal_prior']=g.legal.cumsum()-b.legal
b['runs_prior']=g.tot.cumsum()-b.tot
b['wkts_prior']=g.wkt.cumsum()-b.wkt
b['balls_left']=120-b.legal_prior
inn1=b[b.innings==1].groupby('match_id').tot.sum()
b['target']=np.where(b.innings==2,b.match_id.map(inn1)+1,np.nan)
# keep only matches with both innings
ok=b.groupby('match_id').innings.nunique()==2
b=b[b.match_id.isin(ok[ok].index)].copy()
b['bat_win']=(b.batting_team_id==b.win_team_id).astype(int)
print('matches',b.match_id.nunique(),'balls',len(b))

def feats(inn,bl,w,rs,tg,ipl):
    inn=np.asarray(inn);bl=np.asarray(bl,float);w=np.asarray(w,float);rs=np.asarray(rs,float);tg=np.asarray(tg,float)
    played=120-bl
    crr=np.where(played>0,rs*6/np.maximum(played,1),0)
    need=np.where(inn==2,tg-rs,0)
    rrr=np.where(inn==2,need*6/np.maximum(bl,1),0)
    proj=np.where(inn==1,rs+bl*np.where(played>0,crr/6,1.3)*0+bl*(1.25),0)  # simple par-ish projection
    return np.column_stack([inn,bl,w,rs,need,np.clip(rrr,0,40),crr,np.where(inn==2,rrr-crr,0),np.asarray(ipl),(10-w)*bl])
X=feats(b.innings,b.balls_left,b.wkts_prior,b.runs_prior,b.target,b.is_ipl)
y=b.bat_win.values; grp=b.match_id.values
oof=np.zeros(len(b))
params=dict(max_iter=250,learning_rate=0.06,max_leaf_nodes=24,min_samples_leaf=200,l2_regularization=1.0,random_state=7)
for tr,te in GroupKFold(5).split(X,y,grp):
    mdl=HistGradientBoostingClassifier(**params).fit(X[tr],y[tr]); oof[te]=mdl.predict_proba(X[te])[:,1]
b['wp']=oof
for i in (1,2):
    k=b.innings==i; print('inn',i,'AUC',round(roc_auc_score(y[k],oof[k]),3),'Brier',round(brier_score_loss(y[k],oof[k]),4))
final=HistGradientBoostingClassifier(**params).fit(X,y)

def wp_of(inn,bl,w,rs,tg,ipl,model):
    inn=np.asarray(inn);bl=np.asarray(bl);w=np.asarray(w);rs=np.asarray(rs);tg=np.asarray(tg,float)
    p=model.predict_proba(feats(inn,np.maximum(bl,1),w,rs,tg,ipl))[:,1]
    ch=(inn==2)
    p=np.where(ch&(rs>=tg),1.0,p)
    p=np.where(ch&(rs<tg)&((bl<=0)|(w>=10)),0.0,p)
    return p
# actual post-ball state
bl2=b.balls_left-b.legal; w2=b.wkts_prior+b.wkt; rs2=b.runs_prior+b.tot
# model for post-state: oof within fold would be ideal; use final model for the next-state, oof for current via same final for consistency
wp0=wp_of(b.innings,b.balls_left,b.wkts_prior,b.runs_prior,b.target,b.is_ipl,final)
wp1=wp_of(b.innings,bl2,w2,rs2,b.target,b.is_ipl,final)
b['wp']=wp0; b['wp_after']=wp1
# last ball of innings 2: truth
last2=(b.innings==2)&((bl2<=0)|(w2>=10)|(rs2>=b.target))
b.loc[last2,'wp_after']=b.loc[last2,'bat_win'].astype(float)
# innings 1 end: state where innings ends -> flip from 2nd innings start perspective is implicit in model; keep model value
b['dwp']=b.wp_after-b.wp
# leverage via simulation
b['phase']=np.where(b.over_ball_num.notna()&(b.overid<=6),0,np.where(b.overid<=15,1,2))
L=b[b.legal==1]
outs=[0,1,2,3,4,6]
probs={}
for ph in (0,1,2):
    s=L[L.phase==ph]; p=[(s.runs.eq(r)&(s.wkt==0)).mean() for r in outs]+[s.wkt.mean()]
    probs[ph]=np.array(p)/np.sum(p)
print({k:v.round(3) for k,v in probs.items()})
lev=np.zeros(len(b)); 
for j,(r,wk) in enumerate([(o,0) for o in outs]+[(0,1)]):
    rs_n=b.runs_prior+r; w_n=b.wkts_prior+wk; bl_n=b.balls_left-1
    p1=wp_of(b.innings,bl_n,w_n,rs_n,b.target,b.is_ipl,final)
    pr=np.array([probs[ph][j] for ph in b.phase])
    lev+=pr*np.abs(p1-wp0)
b['lev']=lev
b.to_pickle('balls.pkl')
print(b[['wp','dwp','lev']].describe().round(4))
print(b.groupby(['innings','phase']).lev.mean().round(4))

# ======== s2_pi.py ========
import pandas as pd, numpy as np
b=pd.read_pickle('balls.pkl')
played=120-b.balls_left
S=((played+1)/120).clip(0,1)**0.75
C=1-np.abs(2*b.wp-1)
rrr=np.where(b.innings==2,(b.target-b.runs_prior)*6/np.maximum(b.balls_left,1),0)
b['rrr']=rrr
sr=((np.clip(rrr,0,30)-6)/10).clip(0,1)
# innings-1 par deficit
b['ballno']=played
par=b[b.innings==1].groupby(['is_ipl','ballno']).runs_prior.mean()
par=par.groupby(level=0).transform(lambda s:s.rolling(9,center=True,min_periods=1).mean())
pp=pd.Series(list(zip(b.is_ipl,b.ballno))).map(par.to_dict()).values
sr1=((pp-b.runs_prior)/35).clip(0,1)
sw=((b.wkts_prior-2)/6).clip(0,1)
Q=np.where(b.innings==2,0.65*sr+0.35*sw,0.5*sr1+0.5*sw)
b['S']=S;b['C']=C;b['Q']=Q
b['PI']=100*S*(0.5*C+0.5*Q)
t75,t50=b.PI.quantile([.75,.5])
print('P50,P75,P90',t50,t75,b.PI.quantile(.9))
b['hp']=(b.PI>=t75); b['ord']=(b.PI<=t50)
print(b.groupby('innings').hp.mean().round(3))
print(b.groupby('overid').hp.mean().round(3).to_string())
print(b.groupby('hp')[['rrr','wkts_prior','balls_left','wp','lev']].mean().round(3))
print(b.groupby('hp').apply(lambda d:pd.Series({'RPB':d.tot.mean(),'dot%':(d.tot[d.legal==1]==0).mean(),'wkt%':d.wkt.mean()})).round(3))
print('corr PI vs lev',b[['PI','lev']].corr(method='spearman').iloc[0,1].round(3), ' PI vs |dwp|',b[['PI']].assign(a=b.dwp.abs()).corr(method='spearman').iloc[0,1].round(3))
print(b.groupby(pd.qcut(b.PI,10,duplicates='drop')).agg(lev=('lev','mean'),absd=('dwp',lambda s:s.abs().mean())).round(4))
b.to_pickle('balls2.pkl')

# ======== s2b.py ========
import pandas as pd, numpy as np
b=pd.read_pickle('balls2.pkl')
b['PI']=np.where(b.innings==2,100*b.S*(0.5*b.C+0.5*b.Q),100*b.S*b.Q)
for q in (.7,.75,.8):
    t=b.PI.quantile(q); h=b.PI>=t
    print(q,round(t,1),'inn1 share',round(h[b.innings==1].mean(),3),'inn2',round(h[b.innings==2].mean(),3),'n',h.sum())
t75,t50=b.PI.quantile([.75,.5]); b['hp']=b.PI>=t75; b['ord']=b.PI<=t50
print(b.groupby('hp')[['rrr','wkts_prior','balls_left','wp','lev']].mean().round(3))
print(b.groupby(['innings','hp']).apply(lambda d:pd.Series({'RPB':d.tot.mean(),'wkt%':d.wkt.mean(),'abs_dwp':d.dwp.abs().mean()})).round(4))
print(b.groupby(pd.qcut(b.PI,10,duplicates='drop')).agg(lev=('lev','mean'),absd=('dwp',lambda s:s.abs().mean())).round(4))
print(b.groupby(['innings','overid']).hp.mean().unstack(0).round(2).T.to_string())
b.to_pickle('balls3.pkl'); print(t50,t75)

# ======== s3_rate.py ========
import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings('ignore')
U='data/raw/'
b=pd.read_pickle('balls3.pkl')
teams=pd.read_csv(U+'teams.csv')[['team_id','team_name']].drop_duplicates('team_id').set_index('team_id').team_name
full={'India','Australia','England','South Africa','New Zealand','Pakistan','Sri Lanka','West Indies','Bangladesh','Afghanistan','Ireland','Zimbabwe'}
def tier1(tid):
    n=teams.get(tid,'')
    return 1.0 if (n in full or n in set(teams[teams.index.isin(b[b.is_ipl==1].batting_team_id)])) else 0.5
tm={t:tier1(t) for t in set(b.batting_team_id)|set(b.bowling_team_id)}
b['w_bat']=b.bowling_team_id.map(tm)   # opposition quality for batter
b['w_bowl']=b.batting_team_id.map(tm)  # opposition quality for bowler
# context expectations (all balls, so same yardstick for pressure & ordinary)
b['wb']=pd.cut(b.wkts_prior,[-1,2,5,10],labels=False)
ctx=['is_ipl','innings','overid','wb']
b['bo']=pd.cut(b.batting_order,[0,3,5,7,13],labels=False)
b['ps']=b.pace_or_spin.fillna('Pace')
# player-level frames
ro_b=(b.dismissal_type=='RunOut')
bat=b[~((b.wkt==1)&(b.dismissed_player_id!=b.striker_id)&b.dismissed_player_id.notna())].copy()   # drop non-striker run-outs
def addq(d,extra):
    d=d.copy(); d['E']=d.groupby(ctx+[extra]).dwp.transform('mean'); d['q']=(d.dwp-d.E)*100; d['z']=(d.q/(d.lev*100).clip(lower=0.3)).clip(-6,6); return d
bat['pid']=bat.striker_id; bat['pname']=bat.striker_name
bow=b[~ro_b].copy(); bow['pid']=bow.bowler_id; bow['pname']=bow.bowler_name
bat=addq(bat,'bo'); bow=addq(bow,'ps')
bat['pid']=bat.striker_id; bat['pname']=bat.striker_name
bow['pid']=bow.bowler_id; bow['pname']=bow.bowler_name
bow['q']=-bow.q; bow['z']=-bow.z; bow['w']=bow.w_bowl
bat['w']=bat.w_bat
# batting ball value counts legal + no-ball bat balls; exclude wides for batter (not faced) 
bat=bat[bat.extras_type.fillna('')!='wide']
def build(d,kind):
    d=d.copy()
    out=[]
    for pid,g in d.groupby('pid'):
        hp=g[g.hp]; od=g[g.ord]
        if len(hp)==0: continue
        wsum=hp.w.sum()
        out.append(dict(pid=pid,name=g.pname.iloc[0],hp_balls=len(hp),hp_w=wsum,
            hp_q=(hp.q*hp.w).sum()/wsum, hp_z=(hp.z*hp.w).sum()/wsum,
            od_balls=len(od), od_z=(od.z*od.w).sum()/max(od.w.sum(),1e-9) if len(od) else np.nan,
            hp_matches=hp.match_id.nunique(), hp_runs=hp.runs.sum() if kind=='bat' else hp.tot.sum()+0,
            hp_legal=hp.legal.sum(), hp_w_wk=hp.wkt.sum(), hp_dots=((hp.tot==0)&(hp.legal==1)).sum(),
            hp_bnd=((hp.is_four|hp.is_six)).sum(), hp_runs_tot=hp.tot.sum(),
            all_balls=len(g)))
    return pd.DataFrame(out)
B=build(bat,'bat'); W=build(bow,'bow')
# per-innings consistency
def consist(d,minb=6):
    h=d[d.hp].groupby(['pid','match_id']).agg(n=('q','size'),s=('q','sum'),w=('w','mean')).reset_index()
    h=h[h.n>=minb]; h['pos']=(h.s>0).astype(int)
    return h
cb,cw=consist(bat),consist(bow)
b.to_pickle('balls4.pkl'); bat.to_pickle('bat.pkl'); bow.to_pickle('bow.pkl'); B.to_pickle('B.pkl'); W.to_pickle('W.pkl'); cb.to_pickle('cb.pkl'); cw.to_pickle('cw.pkl')
print(len(B),len(W)); print(B.hp_balls.describe().round(0)); print(W.hp_balls.describe().round(0))
for thr in (30,50,60,80):print('bat elig',thr,(B.hp_balls>=thr).sum(),' bowl elig',thr,(W.hp_balls>=thr).sum())
print(cb.groupby('pid').size().describe().round(1)); print(cw.groupby('pid').size().describe().round(1))
print('mean hp q bat', (bat[bat.hp].q).mean().round(4),'bow',(bow[bow.hp].q).mean().round(4))

# ======== s7_rv.py ========
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
bat=pd.read_pickle('bat.pkl'); bow=pd.read_pickle('bow.pkl'); b=pd.read_pickle('balls4.pkl')
# runs-equivalent of a wicket by innings x phase from regression of dwp on runs and wicket
lam={}
for (i,ph),g in b.groupby(['innings','phase']):
    X=np.column_stack([np.ones(len(g)),g.tot,g.wkt,g.legal]); beta=np.linalg.lstsq(X,g.dwp.values,rcond=None)[0]
    lam[(i,ph)]=-beta[2]/beta[1]
print({k:round(v,1) for k,v in lam.items()})
ctx=['is_ipl','innings','overid','wb']
def addrv(d,extra):
    d=d.copy()
    d['lam']=[lam[(i,p)] for i,p in zip(d.innings,d.phase)]
    d['raw']=d.tot-d.lam*d.wkt
    d['E']=d.groupby(ctx+[extra]).raw.transform('mean')
    d['rv']=d.raw-d.E
    return d
bat=addrv(bat,'bo'); bow=addrv(bow,'ps'); bow['rv']=-bow.rv
bat.to_pickle('bat2.pkl'); bow.to_pickle('bow2.pkl')
for kind,d in (('bat',bat),('bowl',bow)):
    ms=np.array(sorted(set(d.match_id))); rng=np.random.default_rng(3); R=[]
    for rep in range(40):
        perm=rng.permutation(ms); A=set(perm[:len(ms)//2]); 
        comp=lambda x: x[x.hp].groupby('pid').agg(n=('rv','size'),h=('rv','mean')).join(x[x.ord].groupby('pid').agg(no=('rv','size'),o=('rv','mean')))
        ca,cb=comp(d[d.match_id.isin(A)]),comp(d[~d.match_id.isin(A)])
        j=[p for p in ca.index.intersection(cb.index) if ca.n[p]>=30 and cb.n[p]>=30 and ca.no.get(p,0)>=30]
        a,c=ca.loc[j],cb.loc[j]
        R.append(dict(n=len(j),hp_hp=spearmanr(a.h,c.h)[0],base_base=spearmanr(a.o,c.o)[0],base_to_hp=spearmanr(a.o,c.h)[0]))
    print(kind,pd.DataFrame(R).mean().round(3).to_dict())

# ======== s8_final.py ========
import pandas as pd, numpy as np, pickle, json, warnings; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
U='data/raw/'
b=pd.read_pickle('balls4.pkl'); bat=pd.read_pickle('bat2.pkl'); bow=pd.read_pickle('bow2.pkl')
A_C=3; WTS=(.65,.0,.35)
def tables(d,hpm,odm):
    d=d.assign(rw=d.rv*d.w)
    f=lambda m:d[m].groupby(['pid','match_id']).agg(n=('rv','size'),w=('w','sum'),rw=('rw','sum')).reset_index()
    return f(hpm),f(odm)
def hyper(hp,od,el):
    # per-player means, pooled sigma2, tau2 by moments, regression of hp on baseline
    H=hp[hp.pid.isin(el)].groupby('pid').agg(w=('w','sum'),rw=('rw','sum'),n=('n','sum')); O=od[od.pid.isin(el)].groupby('pid').agg(w=('w','sum'),rw=('rw','sum'))
    xh=H.rw/H.w; xo=O.rw/O.w
    return H,O,xh,xo
def run(d,mb=120,mm=10,mo=60,hp_mask=None,mult=None,fixed=None,elig=None,wts=WTS):
    hpm=d.hp if hp_mask is None else hp_mask
    hp,od=tables(d,hpm,d.ord) if hp_mask is None else tables(d,hp_mask,~hp_mask & d.ord)
    if mult is not None:
        for t in (hp,od):
            m=t.match_id.map(mult).fillna(0); t['n']*=m; t['w']*=m; t['rw']*=m; t['m']=m
    else:
        hp['m']=1; od['m']=1
    if elig is None:
        n=hp.groupby('pid').agg(b=('n','sum'),k=('n',lambda s:(s>=6).sum())); ob=od.groupby('pid').n.sum()
        elig=set(n[(n.b>=mb)&(n.k>=mm)].index)&set(ob[ob>=mo].index)
    H=hp[hp.pid.isin(elig)].groupby('pid').agg(w=('w','sum'),rw=('rw','sum'),n=('n','sum'))
    O=od[od.pid.isin(elig)].groupby('pid').agg(w=('w','sum'),rw=('rw','sum'))
    H=H[H.w>0]
    xh=H.rw/H.w; xo=(O.rw/O.w).reindex(H.index); xo=xo.fillna(xo.mean())
    wo=O.w.reindex(H.index).fillna(1.0)
    obs=xh-xo; vr=d.loc[hpm,'rv'].var()*(1/H.w+1/wo)
    if fixed is None:
        s2=d.loc[hpm,'rv'].var()
        tau2=max(xh.var()-(s2/H.w).mean(),1e-4)
        x=np.polyfit(xo,xh,1,w=np.sqrt(H.w)) 
        ms=obs.mean(); ts2=max(obs.var()-vr.mean(),1e-4)
        fixed=dict(s2=s2,tau2=tau2,k=s2/tau2,slope=x[0],icpt=x[1],ms=ms,ts2=ts2)
    # prior from own baseline (usual level), shrunk by reliability of baseline
    prior=fixed['icpt']+fixed['slope']*xo
    post=(H.rw+fixed['k']*prior)/(H.w+fixed['k'])
    # ordinary posterior toward peer mean
    shift=fixed['ms']+fixed['ts2']/(fixed['ts2']+vr)*(obs-fixed['ms'])
    h=hp[(hp.pid.isin(elig))&(hp.n>=6)]
    pos=h.assign(p=((h.rw>0)*h.m)).groupby('pid').agg(p=('p','sum'),kk=('m','sum')).reindex(H.index).fillna(0)
    cons=(pos.p+A_C)/(pos.kk+2*A_C)
    df=pd.DataFrame({'rating':post*100,'shift':shift*100,'consist':cons,'hp_balls':H.n,'raw_hp':xh*100,'baseline':xo*100})
    z=lambda s:(s-s.mean())/s.std()
    df['score']=wts[0]*z(df.rating)+wts[1]*z(df['shift'])+wts[2]*z(df.consist)
    return df.sort_values('score',ascending=False),fixed,elig
out={}
rng=np.random.default_rng(21)
for kind,d in (('bat',bat),('bowl',bow)):
    full,fx,el=run(d)
    full['name']=d.groupby('pid').pname.first().reindex(full.index)
    print(kind,len(el),{k:round(v,4) for k,v in fx.items()})
    ms=np.array(sorted(set(d.match_id))); R=[]
    for i in range(300):
        s=rng.choice(ms,len(ms),replace=True); mult=pd.Series(s).value_counts()
        R.append(run(d,mult=mult,fixed=fx,elig=el)[0]['score'].rank(ascending=False).reindex(full.index))
    R=pd.concat(R,axis=1)
    full['p_top10']=(R<=10).mean(axis=1); full['rank_med']=R.median(axis=1); full['rank_lo']=R.quantile(.05,axis=1); full['rank_hi']=R.quantile(.95,axis=1)
    # reliability: random split halves, 30 reps, score vs score and score(A) vs pressure rv(B)
    rel=[];pred=[]
    for rep in range(30):
        perm=rng.permutation(ms); Aset=set(perm[:len(ms)//2]); mA=pd.Series(1,index=list(Aset)); mB=pd.Series(1,index=list(set(ms)-Aset))
        a=run(d,mult=mA,fixed=fx,elig=el)[0]; c=run(d,mult=mB,fixed=fx,elig=el)[0]
        j=a.index.intersection(c.index); rel.append(spearmanr(a.loc[j,'score'],c.loc[j,'score'])[0]); pred.append(spearmanr(a.loc[j,'rating'],c.loc[j,'raw_hp'])[0])
    # sensitivity
    top=set(full.index[:10]); sens={}
    for nm,w in {'rating_only':(1,0,0),'50_0_50':(.5,0,.5),'80_0_20':(.8,0,.2),'50_25_25':(.5,.25,.25)}.items():
        sens[nm]=len(set(run(d,fixed=fx,elig=el,wts=w)[0].index[:10])&top)/10
    for q in (.70,.80):
        m2=d.PI>=b.PI.quantile(q); t=run(d,hp_mask=m2,mb=120*(1.2 if q==.7 else .8))[0]
        sens[f'pressure_top{int(round((1-q)*100))}pct']=len(set(t.index[:10])&top)/10
    print(kind,'split-half score r',round(np.mean(rel),2),'rating(A)->raw pressure(B)',round(np.mean(pred),2),sens)
    print(full.head(12)[['name','score','rating','shift','consist','hp_balls','p_top10','rank_med']].round(2).to_string())
    out[kind]=dict(full=full,fx=fx,n_elig=len(el),rel=float(np.mean(rel)),pred=float(np.mean(pred)),sens=sens)
pickle.dump(out,open('final2.pkl','wb'))

# ======== s9_export.py ========
import pandas as pd, numpy as np, pickle, json, warnings; warnings.filterwarnings('ignore')
U='data/raw/'
F=pickle.load(open('final2.pkl','rb')); b=pd.read_pickle('balls4.pkl'); bat=pd.read_pickle('bat2.pkl'); bow=pd.read_pickle('bow2.pkl')
mp=pd.read_csv(U+'competition_team_player_mapping.csv'); comps=pd.read_csv(U+'competitions.csv')
mp=mp.merge(comps[['comp_id','match_format','season']],on='comp_id'); mp['yr']=mp.season.astype(str).str[:4].astype(int)
ipl=mp[mp.comp_name.str.startswith('IPL')].sort_values('yr'); intl=mp[mp.match_format=='T20I'].sort_values('yr')
lat_ipl=ipl.groupby('player_id').tail(1).set_index('player_id'); lat_int=intl.groupby('player_id').tail(1).set_index('player_id')
def team(pid):
    if pid in lat_ipl.index: return lat_ipl.loc[pid,'team_name'], int(lat_ipl.loc[pid,'yr'])
    if pid in lat_int.index: return lat_int.loc[pid,'team_name'], int(lat_int.loc[pid,'yr'])
    return 'NA',0
def country(pid): return lat_int.loc[pid,'team_name'] if pid in lat_int.index else ''
for d in (bat,bow): d['ex']=np.where(d.extras_type.fillna('').isin(['wide','Overstep','other']),1,0)
def stats(kind,pid):
    if kind=='bat':
        h=bat[(bat.pid==pid)&bat.hp]; o=bat[(bat.pid==pid)&bat.ord]
        f=lambda x:dict(balls=len(x),runs=int(x.runs.sum()),sr=round(x.runs.sum()/max(len(x),1)*100,1),bnd=round((x.is_four|x.is_six).mean()*100,1),dot=round((x.runs==0).mean()*100,1),outs=int(((x.wkt==1)&(x.dismissed_player_id==pid)).sum()))
        return f(h),f(o)
    h=bow[(bow.pid==pid)&bow.hp&(bow.legal==1)]; o=bow[(bow.pid==pid)&bow.ord&(bow.legal==1)]
    f=lambda x:dict(balls=len(x),runs=int((x.runs+x.ex).sum()),econ=round((x.runs+x.ex).sum()/max(len(x),1)*6,2),dot=round((x.tot==0).mean()*100,1),wkts=int(x.wkt.sum()),bnd=round((x.is_four|x.is_six).mean()*100,1))
    return f(h),f(o)
out={}
for kind in ('bat','bowl'):
    full=F[kind]['full'].copy()
    rows=[]
    for r,(pid,x) in enumerate(full.iterrows(),1):
        t,yr=team(pid); hs,os_=stats(kind,pid)
        rows.append(dict(rank=r,pid=int(pid),name=x['name'],team=t,country=country(pid),score=round(x.score,3),rating=round(x.rating,2),consist=round(x.consist*100,1),
           hp_balls=int(x.hp_balls),p_top10=round(x.p_top10*100),rank_lo=int(x.rank_lo),rank_hi=int(x.rank_hi),press=hs,ordn=os_))
    out[kind]=rows
    print(kind); print(pd.DataFrame(rows).head(10)[['rank','name','team','score','rating','consist','hp_balls','p_top10']].to_string())
# pressure-index evidence
hp=b[b.hp]; od=b[b.ord]
ev={}
for nm,g in (('High pressure',hp),('Ordinary',od),('All balls',b)):
    chase=g[g.innings==2]
    ev[nm]=dict(balls=len(g),rrr=round(chase.rrr.clip(upper=36).median(),2),wk=round(g.wkts_prior.mean(),2),bl=round(g.balls_left.mean(),1),wp_close=round((1-np.abs(2*g.wp-1)).mean(),2),
      rpb=round(g.tot.mean(),3),dot=round(((g.tot==0)&(g.legal==1)).sum()/g.legal.sum()*100,1),wkt=round(g.wkt.mean()*100,2),absd=round(g.dwp.abs().mean()*100,2),lev=round(g.lev.mean()*100,2),inn2share=round((g.innings==2).mean()*100,1))
print(ev)
hpover=b.groupby(['innings','overid']).hp.mean().unstack(0).round(3)
dec=b.groupby(pd.qcut(b.PI,10,labels=False,duplicates='drop')).agg(absd=('dwp',lambda s:s.abs().mean()*100),wkt=('wkt',lambda s:s.mean()*100),dot=('tot',lambda s:(s==0).mean()*100)).round(2)
meta=dict(matches=int(b.match_id.nunique()),balls=int(len(b)),ipl_matches=int(b[b.is_ipl==1].match_id.nunique()),t20i_matches=int(b[b.is_ipl==0].match_id.nunique()),
  hp_balls=int(b.hp.sum()),thr_hp=round(float(b.PI.quantile(.75)),1),thr_ord=round(float(b.PI.quantile(.5)),1),
  auc1=0.683,auc2=0.889,n_elig_bat=F['bat']['n_elig'],n_elig_bowl=F['bowl']['n_elig'],
  rel_bat=round(F['bat']['rel'],2),rel_bowl=round(F['bowl']['rel'],2),sens_bat=F['bat']['sens'],sens_bowl=F['bowl']['sens'],
  k_bat=round(F['bat']['fx']['k']),k_bowl=round(F['bowl']['fx']['k']))
json.dump(dict(bat=out['bat'],bowl=out['bowl'],ev=ev,hpover={str(k):v for k,v in hpover.to_dict().items()},dec=dec.reset_index().to_dict('list'),meta=meta),open('results/data.json','w'))
for k in ('bat','bowl'):
    pd.DataFrame(out[k]).drop(columns=['press','ordn']).to_csv(f'results/{k}_ratings_all_eligible.csv',index=False)
top=pd.concat([pd.DataFrame(out['bat'][:10]).assign(category='Batter'),pd.DataFrame(out['bowl'][:10]).assign(category='Bowler')])[['category','rank','name','team','score','rating','consist','hp_balls','p_top10']]
top.to_csv('results/top10_submission.csv',index=False)
print(meta)
