"""Portable inference reader. No teachers, training caches or workstation imports."""
import json
from pathlib import Path
import numpy as np
import torch
from .models import ProgramDual, GeneProgramDual


class Inputs:
    def __init__(self, assets, bank, device):
        self.root = Path(assets)
        self.device = torch.device(device)
        self.bank = bank
        self.axes = json.loads((self.root/'atlas/axes.json').read_text())
        self.lookup = {t:i for i,t in enumerate(self.axes['targets'])}
        self.native = {t:i for i,t in enumerate(self.axes['native_targets'])}
        self.arrays = {k:np.load(self.root/bank/f'{k}.npy', mmap_mode='r')
                       for k in ('references','response','controls','cells','mask')}
        self.esm = np.load(self.root/'atlas/esm.npy')
        self.effect = self.tensor(np.load(self.root/'atlas/evidence.npy'))
        self.evidence_mask = self.tensor(np.load(self.root/'atlas/evidence_mask.npy'))
        with np.load(self.root/'atlas/native_cd4.npz') as z:
            self.cd4 = self.tensor(z['effect'], torch.float64)
            self.available = self.tensor(z['available'], torch.float64)
        self.promoter = torch.ones((len(self.native),len(self.axes['genes'])),
                                   device=self.device, dtype=torch.float64)
        self.promoter_mask = torch.zeros_like(self.promoter, dtype=torch.bool)
        import pandas as pd
        columns = {g:i for i,g in enumerate(self.axes['genes'])}
        for r in pd.read_csv(self.root/'atlas/promoters.csv').itertuples(index=False):
            if r.target not in self.native or r.neighbor not in columns or r.target == r.neighbor:
                continue
            if not 0 <= r.distance <= 5000: raise ValueError('Invalid promoter distance')
            i,j = self.native[r.target],columns[r.neighbor]
            remaining = .15+.85*np.clip(np.log(max(float(r.distance),500)/500)/np.log(10),0,1)
            self.promoter[i,j] = min(float(self.promoter[i,j]),remaining)
            self.promoter_mask[i,j] = True

    def tensor(self, value, dtype=None):
        return torch.tensor(np.asarray(value), device=self.device, dtype=dtype)

    @torch.no_grad()
    def fields(self, features, measured, targets):
        ids = np.array([self.lookup[t] for t in targets])
        x = self.tensor(features, torch.float32)
        mask = self.tensor(measured, torch.bool).expand(len(ids),-1)
        control = x[[1,0]].expand(len(ids),-1,-1)
        mean = torch.expm1(control[:,0].double())/50000
        q = torch.expm1(x[0].double())/50000; q /= q.sum()
        q = q.expand(len(ids),-1)
        prior = torch.log1p(50000*q).float()
        selected = [i for i,t in enumerate(targets) if t in self.native]
        if selected:
            pos = self.tensor(selected, torch.long)
            native = self.tensor([self.native[targets[i]] for i in selected], torch.long)
            weights = self.evidence_mask[native].double()*q.new_tensor([2,1,1,2])[None,:,None]
            available = self.available[native]
            den = weights.sum(1)+.5*available
            num = (self.effect[native,:,:,1].double()*weights).sum(1)
            num += .5*self.cd4[native]*available
            effect = torch.where(den>0,num/den.clamp_min(1e-12),0)
            p = mean[pos]*torch.exp2((.6*effect).clamp(-3,3)); p /= p.sum(-1,keepdim=True)
            p = torch.where(self.promoter_mask[native],torch.minimum(p,mean[pos]*self.promoter[native]),p)
            mean[pos] = p/p.sum(-1,keepdim=True)
            transferred = torch.log1p(50000*q[pos]*torch.exp2(self.cd4[native].clamp(-10,10)))-torch.log1p(50000*q[pos])
            num = (self.effect[native,:,:,0].double()*weights).sum(1)
            num += .5*torch.where(available>0,transferred,0)
            delta = torch.where(den>0,num/den.clamp_min(1e-12),0)
            p = torch.expm1((torch.log1p(50000*q[pos])+(.3*delta).clamp(-3,3)).clamp_min(0))
            p /= p.sum(-1,keepdim=True)
            p = torch.where(self.promoter_mask[native],torch.minimum(p,q[pos]*self.promoter[native]),p)
            p /= p.sum(-1,keepdim=True); p *= mask[pos]; p /= p.sum(-1,keepdim=True)
            prior[pos] = torch.log1p(50000*p).float()
        mean *= mask; mean /= mean.sum(-1,keepdim=True)
        mass = torch.expm1(prior.double())/50000*mask
        mass = (mass/mass.sum(-1,keepdim=True)+1e-12)*mask
        bulk = (mass/mass.sum(-1,keepdim=True)).float()
        refs = self.arrays['references'][ids]
        valid = self.tensor(refs>=0)
        safe = np.maximum(refs,0)
        args = (torch.stack((mean.float(),bulk),1),control,self.tensor(self.esm[ids]),
            self.tensor(self.arrays['response'][safe]),self.tensor(self.arrays['controls'][safe]),
            self.tensor(self.arrays['cells'][safe],torch.float32)*valid,
            self.tensor(self.arrays['mask'][safe]) & valid[...,None],valid)
        if self.bank == 'gene':
            return (*args,self.tensor(np.array(self.axes['target_ids'])[ids],torch.long),mask)
        return (*args,mask)

    @torch.inference_mode()
    def predict(self, checkpoint, features, measured, targets):
        if len(set(targets)) != len(targets) or not set(targets) <= set(self.lookup):
            raise ValueError('Repeated/unsupported targets; extend the source bank explicitly')
        if features.shape != (4,len(self.axes['genes'])) or measured.shape != features.shape[1:]:
            raise ValueError('Control gene axis differs')
        if not np.isfinite(features).all() or (features<0).any() or features[:,~measured].any():
            raise ValueError('Invalid control statistics')
        np.testing.assert_allclose((np.expm1(features[:2].astype(float))/50000).sum(1),1,rtol=0,atol=5e-7)
        cls = GeneProgramDual if self.bank == 'gene' else ProgramDual
        kwargs = dict(targets=self.axes['training_target_count']) if self.bank == 'gene' else {}
        model = cls(len(self.axes['genes']),self.esm.shape[1],**kwargs).to(self.device)
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        model.eval(); out = {k:[] for k in ('probability','weights','base')}
        for start in range(0,len(targets),64):
            args = self.fields(features,measured,targets[start:start+64])
            with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
                q,w = model(*args)
            if not torch.isfinite(q).all() or (q<0).any() or torch.count_nonzero(q*~args[-1][:,None]):
                raise ValueError('Invalid neural prediction')
            if self.device.type=='cpu':
                # CPU softmax's long-axis accumulation differs; keep strict downstream unit-mass guards.
                mass=q.double(); q=(mass/mass.sum(-1,keepdim=True)).float()
            torch.testing.assert_close(q.sum(-1),torch.ones_like(q.sum(-1)),rtol=0,atol=5e-7)
            for key,value in zip(out,(q,w,args[0])): out[key].append(value.cpu().numpy())
        del model
        return {k:np.concatenate(v) for k,v in out.items()}
