"""Original full-axis training objectives, independent of inference readout."""
import torch


def objective(probability, teacher, weight, own_columns, bulk_coefficient=0.):
    ce=-(teacher.float()*probability.float().clamp_min(1e-30).log()).sum(-1)
    ce=(ce*weight[:,None]).mean(0)
    error=(torch.log1p(50000*probability[:,1].float())-
           torch.log1p(50000*teacher[:,1].float())).square()
    own_columns=own_columns.to(device=error.device,dtype=torch.long)
    present=own_columns>=0
    own=error.gather(1,own_columns.clamp_min(0)[:,None]).squeeze(1)*present
    per_group=(error.sum(-1)-own)/(error.shape[-1]-present.to(error.dtype))
    mse=(per_group*weight.float()).mean()
    total=ce.mean()+bulk_coefficient*mse
    return total,dict(mean_CE=ce[0],bulk_CE=ce[1],bulk_log_MSE=mse,
                      bulk_log_contribution=bulk_coefficient*mse)
