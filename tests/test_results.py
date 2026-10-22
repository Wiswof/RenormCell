import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd


@unittest.skipUnless(importlib.util.find_spec('matplotlib'),'Install the optional plots extra')
class ResultsTests(unittest.TestCase):
    def test_normalized_mse_uses_ratio_of_sums(self):
        from scripts.plot_results import aggregate
        t=pd.DataFrame({'expr_mse_unbiased_capped':[1.,3.],'expr_distance_unbiased':[1.,9.]})
        value,count=aggregate(t,'expr_mse_unbiased_capped_norm',np.arange(2))
        self.assertAlmostEqual(float(value),.4);self.assertEqual(count,2)

    def test_every_published_local_aggregate_matches_export(self):
        from scripts.plot_results import aggregate
        root=Path(__file__).resolve().parents[1]/'results'
        targets=pd.read_csv(root/'per_target_metrics.csv')
        reported=pd.read_csv(root/'local_aggregates.csv')
        self.assertEqual(len(reported),54)
        for (scope,profile),rows in reported.groupby(['scope','profile']):
            selected=targets[(targets.scope==scope)&(targets.profile==profile)]
            table=selected.pivot(index='perturbation',columns='metric',values='value')
            for row in rows.itertuples(index=False):
                value,_=aggregate(table,row.metric,np.arange(len(table)))
                self.assertAlmostEqual(float(value),row.value,places=10)


if __name__=='__main__':unittest.main()
