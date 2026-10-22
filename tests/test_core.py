import tempfile
import unittest
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import torch
from renormcell.controls import population_features
from renormcell.models import ProgramDual,GeneProgramDual
from renormcell.delta import counts as dual_counts
from renormcell.bulk import counts as bulk_counts
from renormcell.readout import endpoints
from renormcell.io import CountWriter
from renormcell.prepare import prepare
from renormcell.losses import objective


class CoreTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def test_two_means_are_different(self):
        x=sparse.csr_matrix([[1,9],[90,10]])
        f=population_features(x)
        np.testing.assert_array_equal(f[:2],
            np.log1p(50000*np.array([[91/110,19/110],[.5,.5]])).astype(np.float32))
        with self.assertRaises(ValueError): population_features(sparse.csr_matrix([[.2,1.]]))

    def test_both_model_axes_and_missing_source_masks(self):
        torch.manual_seed(7)
        batch,genes,sources=2,8,3
        mask=torch.tensor([[True]*6+[False]*2]*batch)
        base=mask.float()/6;base=torch.stack((base,base),1)
        args=(base,torch.zeros(batch,2,genes),torch.randn(batch,5),
            torch.randn(batch,sources,2,genes),torch.zeros(batch,sources,2,genes),
            torch.ones(batch,sources)*100,mask[:,None].expand(-1,sources,-1),
            torch.tensor([[True,False,True]]*batch))
        for cls in (ProgramDual,GeneProgramDual):
            model=cls(genes,5,width=8,rank=4,**(dict(hidden=16,targets=4) if cls==GeneProgramDual else {}))
            fields=(*args,torch.tensor([0,1]),mask) if cls==GeneProgramDual else (*args,mask)
            q,_=model(*fields)
            self.assertEqual(q.shape,(batch,2,genes))
            self.assertTrue(torch.isfinite(q).all());self.assertEqual(int(torch.count_nonzero(q*~mask[:,None])),0)
            torch.testing.assert_close(q.sum(-1),torch.ones(batch,2))

    def test_zero_correction_is_exact(self):
        x=np.random.default_rng(4).integers(1,20,(16,8));q=np.ones((2,8))/8
        for function in (dual_counts,bulk_counts):
            y,detail=function(x,q,q,np.ones(8,bool),1)
            np.testing.assert_array_equal(x,y);self.assertTrue(detail['zero_delta_exact_scaffold'])

    def test_full_panel_centering_and_identity(self):
        q=np.ones((4,2,8))/8
        np.testing.assert_array_equal(endpoints(q,q,'target_delta'),q)
        trained=q.copy();trained[0,:,0]*=2;trained/=trained.sum(-1,keepdims=True)
        result=endpoints(q,trained,'target_delta')
        np.testing.assert_allclose(result.sum(-1),1)
        self.assertFalse(np.array_equal(result[1:],q[1:]))

    def test_streaming_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'counts.h5ad'
            with CountWriter(path,['T'],['a','b'],['A'],3) as writer:
                writer.append(np.array([[1,2],[2,4],[0,1]]))
            a=ad.read_h5ad(path)
            self.assertEqual(a.shape,(3,2));self.assertEqual(a.obs.target_gene.tolist(),['T']*3)
            np.testing.assert_array_equal(a.X.toarray(),[[1,2],[2,4],[0,1]])

    def test_preparation_requires_explicit_calls_and_preserves_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);rng=np.random.default_rng(9)
            x=rng.integers(1,8,(50,3))
            obs=pd.DataFrame(dict(study='test',cell_line='line',context='batch',
                target_gene=['NTC']*30+['T']*20,guide=['NT']*30+['sgT']*20,assay='CRISPRi',
                is_control=[True]*30+[False]*20,quality_pass=True),index=[f'c{i}' for i in range(50)])
            a=ad.AnnData(sparse.csr_matrix(x),obs=obs,var=pd.DataFrame(index=['a','b','c']))
            a.write_h5ad(tmp/'input.h5ad');prepare(tmp/'input.h5ad',['a','b','c','missing'],tmp/'out')
            target=population_features(sparse.csr_matrix(np.column_stack((x[30:],np.zeros(20,int)))))[[1,0]]
            control=population_features(sparse.csr_matrix(np.column_stack((x[:30],np.zeros(30,int)))))[[1,0]]
            np.testing.assert_array_equal(np.load(tmp/'out/response.npy')[0],target-control)
            np.testing.assert_array_equal(np.load(tmp/'out/mask.npy')[0],[1,1,1,0])
            a.obs['assay']='CRISPRa';a.write_h5ad(tmp/'bad.h5ad')
            with self.assertRaises(ValueError):prepare(tmp/'bad.h5ad',['a','b','c'],tmp/'bad')

    def test_objective_has_explicit_mean_bulk_and_mse_terms(self):
        q=torch.ones(2,2,4)/4;t=q.clone()
        total,parts=objective(q,t,torch.ones(2),torch.tensor([0,-1]),.2853516127573863)
        self.assertAlmostEqual(float(total),float(torch.log(torch.tensor(4.))),places=6)
        self.assertEqual(float(parts['bulk_log_MSE']),0.)


if __name__=='__main__':unittest.main()
