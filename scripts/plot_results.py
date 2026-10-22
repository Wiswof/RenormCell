"""Redraw published six-metric figures from the accompanying CSVs; no model/GPU needed."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS={
    'pds_cosine':'PDS cosine  (higher better)',
    'expr_mse_unbiased_capped_norm':'Normalized MSE  (lower better)',
    'de_wilcoxon_lfc_nmae':'DE log-FC NMAE  (lower better)',
    'de_wilcoxon_direction_fidelity_yield_raw':'Direction fidelity  (higher better)',
    'de_wilcoxon_direction_reach_raw':'Magnitude reach  (higher better)',
    'de_wilcoxon_sig_jaccard':'DE Jaccard  (higher better)',
}
PROFILES=('Atlas+K448','official-best','final-candidate')
COLORS=('#777777','#0072B2','#D55E00')
SHORT=('Atlas+K448','official-best','final-candidate')


def aggregate(table,metric,indices):
    """Match SDK means; normalized MSE is ratio-of-sums, NOT mean of target ratios."""
    if metric=='expr_mse_unbiased_capped_norm':
        a=table['expr_mse_unbiased_capped'].to_numpy(float)
        b=table['expr_distance_unbiased'].to_numpy(float)
        valid=np.isfinite(a)&np.isfinite(b)
        numerator=np.where(valid,a,0)[indices].sum(axis=-1)
        denominator=np.where(valid,b,0)[indices].sum(axis=-1)
    else:
        a=table[metric].to_numpy(float)
        if np.isinf(a).any():raise ValueError('Infinite target metric')
        valid=np.isfinite(a)
        numerator=np.where(valid,a,0)[indices].sum(axis=-1)
        denominator=valid[indices].sum(axis=-1)
    result=np.full(np.shape(numerator),np.nan)
    np.divide(numerator,denominator,out=result,where=denominator>0)
    return result,int(valid.sum())


def local_figures(root):
    targets=pd.read_csv(root/'per_target_metrics.csv')
    reported=pd.read_csv(root/'local_aggregates.csv')
    rows=[]
    for index,scope in enumerate(('h1','kolf','dld')):
        data=targets[targets.scope==scope]
        tables={p:data[data.profile==p].pivot(index='perturbation',columns='metric',values='value').sort_index()
                for p in PROFILES}
        first=tables[PROFILES[0]]
        if any(not t.index.equals(first.index) for t in tables.values()):raise ValueError('Unpaired target panels')
        draws=np.random.default_rng(20261017+index).integers(len(first),size=(2000,len(first)))
        fig,axes=plt.subplots(2,3,figsize=(15,8.4))
        for ax,(metric,title) in zip(axes.flat,METRICS.items()):
            labels=[]
            for x,(profile,color) in enumerate(zip(PROFILES,COLORS)):
                table=tables[profile]
                value,count=aggregate(table,metric,np.arange(len(table)))
                expected=reported[(reported.scope==scope)&(reported.profile==profile)&(reported.metric==metric)].value
                if len(expected)!=1:raise ValueError('Missing/duplicate aggregate')
                np.testing.assert_allclose(value,expected.iloc[0],rtol=0,atol=1e-10)
                sample,_=aggregate(table,metric,draws);valid=np.isfinite(sample)
                low,high=(np.quantile(sample[valid],[.025,.975]) if count>=2 and valid.mean()>=.95 else (np.nan,np.nan))
                ax.scatter(x,value,s=65,color=color,zorder=3)
                if np.isfinite([low,high]).all():
                    ax.vlines(x,low,high,color=color,lw=2)
                    ax.plot([x,x],[low,high],'_',color=color,markersize=10,markeredgewidth=2)
                labels.append(f'{SHORT[x]}\n{value:.5g} | n={count}')
                rows.append(dict(scope=scope,profile=profile,metric=metric,value=float(value),
                    ci_low=float(low),ci_high=float(high),finite_targets=count,
                    invalid_bootstrap_fraction=float(1-valid.mean())))
            ax.set_title(title,fontsize=12,pad=11)
            ax.set_xticks(range(3),labels,fontsize=9);ax.set_xlim(-.45,2.45)
            ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True);ax.margins(y=.2)
            ax.ticklabel_format(axis='y',style='plain',useOffset=False)
            ax.spines[['top','right']].set_visible(False)
        names=dict(h1='H1: guide-heldout development',kolf='KOLF: same-study cell holdout',dld='DLD-1: source-isolated scoring panel')
        fig.suptitle(f'{names[scope]} | six raw metrics | {len(first)} targets, 400 generated cells/target',fontsize=15,y=.98)
        fig.text(.5,.025,'95% target bootstrap (2,000 paired resamples); fixed generation seed and PDS panel ranks.\n'
            'Repeated/exposed local development, not independent validation or official competition points.',
            ha='center',fontsize=10,color='#444444')
        fig.tight_layout(rect=(0,.08,1,.94),h_pad=3,w_pad=3)
        fig.savefig(root/f'{scope}_six_metrics.png',dpi=160);plt.close(fig)
    pd.DataFrame(rows).to_csv(root/'local_intervals.csv',index=False)


def official_figure(root):
    data=pd.read_csv(root/'official_validation.csv').set_index('metric')
    metrics=('score_pds','score_mse','score_nmae','score_fid','score_reach','score_jac')
    names=('PDS credit','MSE credit','DE log-FC accuracy credit','Fidelity credit','Magnitude-reach credit','DE Jaccard credit')
    for col in ('previous','current'):
        np.testing.assert_allclose(data.loc[list(metrics),col].mean(),data.loc['score_avg',col],rtol=0,atol=1e-14)
    fig,axes=plt.subplots(2,3,figsize=(15,7.5))
    for ax,m,title in zip(axes.flat,metrics,names):
        values=data.loc[m,['previous','current']].to_numpy(float)
        ax.barh([1,0],values,color=COLORS[:2],height=.55)
        ax.set_yticks([1,0],['Atlas+K448','official-best'],fontsize=10)
        span=max(abs(values).max(),.001);lo=min(0,values.min());hi=max(0,values.max())
        ax.set_xlim(lo-.3*span,hi+.45*span)
        for y,value in zip([1,0],values):
            ax.text(value/2,y,f'{value:.6f}',va='center',ha='center',
                fontsize=10,color='white',fontweight='bold')
        ax.axvline(0,color='#444444',lw=.8)
        ax.set_title(f'{title}\nchange {data.loc[m,"delta"]:+.6f}',fontsize=12)
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    before,after=data.loc['score_avg',['previous','current']]
    fig.suptitle(f'Official validation: {before:.6f} -> {after:.6f}  (change {after-before:+.6f})\n'
                 'Same validation panel/anchors; all six normalized credits are higher-is-better',fontsize=15,y=.98)
    fig.text(.5,.025,'Recorded official aggregate values, not raw metrics; no uncertainty interval is available.\n'
        'final-candidate has no official final score and is intentionally not plotted as zero.',ha='center',fontsize=10,color='#444444')
    fig.tight_layout(rect=(0,.09,1,.9),h_pad=3,w_pad=3)
    fig.savefig(root/'official_six_credits.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path(__file__).resolve().parents[1]/'results')
    root=p.parse_args().results
    local_figures(root);official_figure(root)
    print('RESULT_FIGURES_READY',root,flush=True)


if __name__=='__main__':main()
