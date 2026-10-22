"""Frozen ProgramDual32 and GeneProgramDual64; original parameter names preserved."""
import math
import torch
from torch import nn

def transported_experts(base, control_log, response, measured, observed):
    """Missing source genes retain Atlas mass, rather than becoming observed zeros."""
    support = measured.bool() & observed[..., None]
    value = torch.where(support, response.float(), 0.)
    mass = torch.expm1((control_log[:, None].float()+value).clamp_min(0))*support
    total = mass.sum(-1, keepdim=True)
    available_mass = (base[:, None]*support).sum(-1, keepdim=True)
    valid = observed & (total[..., 0] > 0) & (available_mass[..., 0] > 0)
    probability = torch.where(support, mass/total.clamp_min(1e-20)*available_mass, base[:, None])
    probability = torch.where(valid[..., None], probability, base[:, None])
    return probability, valid


class ExpertMixer(nn.Module):
    def __init__(self, genes, esm_dim, width=64):
        super().__init__()
        self.control = nn.Sequential(nn.Linear(genes, width, bias=False), nn.LayerNorm(width), nn.SiLU())
        self.response = nn.Sequential(nn.Linear(genes, width, bias=False), nn.LayerNorm(width), nn.SiLU())
        self.target = nn.Sequential(nn.LayerNorm(esm_dim), nn.Linear(esm_dim, width), nn.SiLU())
        self.score = nn.Sequential(nn.Linear(3*width+4, width), nn.SiLU(), nn.Linear(width, 1))
        nn.init.zeros_(self.score[-1].weight); nn.init.constant_(self.score[-1].bias, -2.)
        self.atlas_score = nn.Parameter(torch.tensor(3.))

    def forward(self, base, control_log, esm, response, source_control, source_cells, source_mask, observed, query_mask):
        # All source rows must be qualified TRAIN references before calling this model.
        support = source_mask.bool() & query_mask[:, None].bool() & observed[..., None]
        r = torch.where(support, response.float(), 0.)
        difference = torch.where(support, source_control.float()-control_log[:, None].float(), 0.)
        genes = support.sum(-1).clamp_min(1)
        extra = torch.stack((torch.log1p(source_cells.float())/10,
            support.float().mean(-1), (r.square().sum(-1)/genes).sqrt(),
            (difference.square().sum(-1)/genes).sqrt()), -1)
        target = self.target(esm).unsqueeze(1).expand(-1, response.shape[1], -1)
        features = torch.cat((self.control(difference), self.response(r), target, extra), -1)
        source_scores = self.score(features).squeeze(-1).float()
        experts, valid = transported_experts(base, control_log, r, support, observed)
        logits = torch.cat((self.atlas_score.expand(len(base), 1), source_scores.masked_fill(~valid, -torch.inf)), 1)
        weights = logits.softmax(-1)
        probability = weights[:, :1]*base + (weights[:, 1:, None]*experts).sum(1)
        probability = torch.where(valid.any(1, keepdim=True),
            probability/probability.sum(-1, keepdim=True), base)
        return probability, weights


class DualMixer(nn.Module):
    def __init__(self, genes, esm_dim, width=64):
        super().__init__()
        self.heads = nn.ModuleList([ExpertMixer(genes, esm_dim, width) for _ in range(2)])

    def forward(self, base, controls, esm, response, source_controls, cells, masks, valid, measured):
        outputs = [head(base[:, k], controls[:, k], esm, response[:, :, k], source_controls[:, :, k],
                        cells, masks, valid, measured) for k, head in enumerate(self.heads)]
        return torch.stack([q for q, _ in outputs], 1), torch.stack([w for _, w in outputs], 1)



class ProgramExpert(nn.Module):
    def __init__(self, fresh_core, genes, width, rank):
        super().__init__()
        self.core = fresh_core  # Fresh shared-layer initialization, never trained weights.
        self.program = nn.Linear(3*width+4,rank)
        nn.init.zeros_(self.program.weight);nn.init.zeros_(self.program.bias)
        self.gene_programs = nn.Parameter(torch.randn(genes,rank)/math.sqrt(rank))
        self.rank = rank

    def forward(self,base,control,esm,response,source_control,cells,masks,observed,measured):
        core = self.core
        support = masks.bool() & measured[:,None].bool() & observed[...,None]
        r = torch.where(support,response.float(),0.)
        difference = torch.where(support,source_control.float()-control[:,None].float(),0.)
        genes = support.sum(-1).clamp_min(1)
        extra = torch.stack((torch.log1p(cells.float())/10,support.float().mean(-1),
                             (r.square().sum(-1)/genes).sqrt(),(difference.square().sum(-1)/genes).sqrt()),-1)
        target = core.target(esm)[:,None].expand(-1,response.shape[1],-1)
        features = torch.cat((core.control(difference),core.response(r),target,extra),-1)
        scores = core.score(features).float()
        scores = scores+(self.program(features)@self.gene_programs.T).float()/math.sqrt(self.rank)
        experts,valid = transported_experts(base,control,r,support,observed)
        atlas = core.atlas_score.expand(len(base),1,base.shape[-1])
        weights = torch.cat((atlas,scores.masked_fill(~valid[...,None],-torch.inf)),1).softmax(1)
        q = weights[:,0]*base+(weights[:,1:]*experts).sum(1)
        q = torch.where(valid.any(1,keepdim=True),q/q.sum(-1,keepdim=True),base)
        # Gene-dependent weights require this unit-mass normalization: NOT a global convex hull claim.
        summary = (weights*measured[:,None]).sum(-1)/measured.sum(-1,keepdim=True).clamp_min(1)
        return q,summary


class ProgramDual(nn.Module):
    def __init__(self,genes,esm_dim,width=64,rank=32):
        super().__init__()
        original = DualMixer(genes,esm_dim,width)
        self.heads = nn.ModuleList([ProgramExpert(head,genes,width,rank) for head in original.heads])

    def forward(self,base,controls,esm,response,source_controls,cells,masks,valid,measured):
        values = [head(base[:,k],controls[:,k],esm,response[:,:,k],source_controls[:,:,k],cells,masks,valid,measured)
                  for k,head in enumerate(self.heads)]
        return torch.stack([q for q,_ in values],1),torch.stack([w for _,w in values],1)


def projection(inputs,width):
    return nn.Sequential(nn.Linear(inputs,width),nn.LayerNorm(width),nn.SiLU())


class ResidualBlock(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.layers=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,2*width),nn.SiLU(),nn.Linear(2*width,width))

    def forward(self,x):return x+self.layers(x)



class BasisHead(nn.Module):
    def __init__(self,genes,width=128,hidden=512):
        super().__init__()
        self.control=projection(genes,width)
        self.response=projection(genes,width)
        self.difference=projection(genes,width)
        self.source=nn.Sequential(nn.Linear(3*width+4,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.score=nn.Linear(width,1)
        self.body=nn.Sequential(nn.Linear(5*width,hidden),nn.SiLU(),*[ResidualBlock(hidden) for _ in range(4)],nn.LayerNorm(hidden))
        self.coefficient=nn.Sequential(nn.Linear(hidden+width,width),nn.SiLU(),nn.Linear(width,1))
        nn.init.zeros_(self.coefficient[-1].weight);nn.init.zeros_(self.coefficient[-1].bias)

    def forward(self,base,control,target,identity,response,source_control,cells,masks,observed,measured):
        support=masks.bool() & measured[:,None].bool() & observed[...,None]
        r=torch.where(support,response.float(),0.)
        d=torch.where(support,source_control.float()-control[:,None].float(),0.)
        count=support.sum(-1).clamp_min(1)
        extra=torch.stack((torch.log1p(cells.float())/10,support.float().mean(-1),
            (r.square().sum(-1)/count).sqrt(),(d.square().sum(-1)/count).sqrt()),-1)
        encoded=self.source(torch.cat((self.response(r),self.difference(d),
            target[:,None].expand(-1,r.shape[1],-1),extra),-1))
        base_response=(torch.log1p(50000*base.float())-control.float())*measured
        base_extra=torch.stack((torch.zeros_like(count[:,0]),measured.float().mean(-1),
            (base_response.square().sum(-1)/measured.sum(-1).clamp_min(1)).sqrt(),torch.zeros_like(count[:,0])),-1)
        atlas=self.source(torch.cat((self.response(base_response),self.difference(torch.zeros_like(control)),target,base_extra),-1))
        experts,valid=transported_experts(base,control,r,support,observed)
        tokens=torch.cat((atlas[:,None],encoded),1)
        available=torch.cat((torch.ones_like(valid[:,:1]),valid),1)
        attention=self.score(tokens).float().squeeze(-1).masked_fill(~available,-torch.inf).softmax(-1)
        pooled=(tokens*attention[...,None]).sum(1)
        hidden=self.body(torch.cat((self.control(control*measured),self.response(base_response),pooled,target,identity),-1))
        coefficients=self.coefficient(torch.cat((hidden[:,None].expand(-1,r.shape[1],-1),encoded),-1)).float().squeeze(-1)
        coefficients=coefficients.masked_fill(~valid,0.)  # Signed, not softmax probabilities or fixed amplitude bounds.
        basis=experts.float().clamp_min(1e-12).log()-base[:,None].float().clamp_min(1e-12).log()
        delta=(coefficients[...,None]*basis).sum(1)
        logits=base.float().clamp_min(1e-12).log()+delta
        probability=logits.masked_fill(~measured,-torch.inf).softmax(-1)
        # Shared engine calls this field "weights". First column is coefficient L1, NOT an Atlas weight.
        diagnostic=torch.cat((coefficients.abs().sum(-1,keepdim=True),coefficients),-1)
        return probability,diagnostic


class BasisDual(nn.Module):
    def __init__(self,genes,esm_dim,targets=19790,width=128,hidden=512):
        super().__init__()
        self.target=projection(esm_dim,width)
        self.identity=nn.Embedding(targets,width)
        self.heads=nn.ModuleList([BasisHead(genes,width,hidden) for _ in range(2)])

    def forward(self,base,controls,esm,response,source_controls,cells,masks,observed,target_ids,measured):
        target=self.target(esm);identity=self.identity(target_ids)
        results=[head(base[:,k],controls[:,k],target,identity,response[:,:,k],source_controls[:,:,k],cells,masks,observed,measured)
            for k,head in enumerate(self.heads)]
        return torch.stack([q for q,_ in results],1),torch.stack([w for _,w in results],1)


class GeneProgramHead(BasisHead):
    def __init__(self, genes, width=128, hidden=512, rank=64):
        super().__init__(genes, width, hidden)
        self.rank = rank
        self.programs = nn.Parameter(torch.randn(genes, rank)/math.sqrt(rank))
        self.coefficient[-1] = nn.Linear(width, rank+1)
        self.residual = nn.Linear(hidden, rank)
        for layer in (self.coefficient[-1], self.residual):
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

    def forward(self, base, control, target, identity, response, source_control, cells, masks, observed, measured):
        # Preserve the frozen basis model's inputs/encoder, not its single global coefficient restriction.
        support = masks.bool() & measured[:, None].bool() & observed[..., None]
        r = torch.where(support, response.float(), 0.)
        d = torch.where(support, source_control.float()-control[:, None].float(), 0.)
        count = support.sum(-1).clamp_min(1)
        extra = torch.stack((torch.log1p(cells.float())/10, support.float().mean(-1),
            (r.square().sum(-1)/count).sqrt(), (d.square().sum(-1)/count).sqrt()), -1)
        encoded = self.source(torch.cat((self.response(r), self.difference(d),
            target[:, None].expand(-1, r.shape[1], -1), extra), -1))
        base_response = (torch.log1p(50000*base.float())-control.float())*measured
        base_extra = torch.stack((torch.zeros_like(count[:, 0]), measured.float().mean(-1),
            (base_response.square().sum(-1)/measured.sum(-1).clamp_min(1)).sqrt(), torch.zeros_like(count[:, 0])), -1)
        atlas = self.source(torch.cat((self.response(base_response), self.difference(torch.zeros_like(control)), target, base_extra), -1))
        experts, valid = transported_experts(base, control, r, support, observed)
        tokens = torch.cat((atlas[:, None], encoded), 1)
        available = torch.cat((torch.ones_like(valid[:, :1]), valid), 1)
        attention = self.score(tokens).float().squeeze(-1).masked_fill(~available, -torch.inf).softmax(-1)
        pooled = (tokens*attention[..., None]).sum(1)
        hidden = self.body(torch.cat((self.control(control*measured), self.response(base_response), pooled, target, identity), -1))
        code = self.coefficient(torch.cat((hidden[:, None].expand(-1, r.shape[1], -1), encoded), -1))
        code = code.masked_fill(~valid[..., None], 0.)
        gate = code[..., :1].float()+(code[..., 1:]@self.programs.T).float()
        basis = experts.float().clamp_min(1e-12).log()-base[:, None].float().clamp_min(1e-12).log()
        delta = (gate*basis).sum(1)+(self.residual(hidden)@self.programs.T).float()
        logits = base.float().clamp_min(1e-12).log()+delta
        probability = logits.masked_fill(~measured, -torch.inf).softmax(-1)
        rms = ((delta.detach().square()*measured).sum(-1)/measured.sum(-1).clamp_min(1)).sqrt()
        # Legacy engine reads weights[...,0]; here it is logit RMS, NOT source/Atlas attention.
        diagnostic = torch.cat((rms[:, None], code[..., 0].float().detach()), -1)
        return probability, diagnostic


class GeneProgramDual(BasisDual):
    def __init__(self, genes, esm_dim, targets=19790, width=128, hidden=512, rank=64):
        super().__init__(genes, esm_dim, targets, width, hidden)
        self.heads = nn.ModuleList([GeneProgramHead(genes, width, hidden, rank) for _ in range(2)])

