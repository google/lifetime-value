# Copyright 2019 The Lifetime Value Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Lint as: python3
# Dependency imports
import sys
import os
import io
current_path = os.getcwd()
sys.path.append(os.getcwd())
sys.path.append(os.path.join(current_path, "lifetime_value"))
print(sys.path)
from lifetime_value import metrics
import numpy as np
import pandas as pd
import unittest


class MetricsTest(unittest.TestCase):

  def setUp(self):
    super(MetricsTest, self).setUp()
    n_example = 1000000
    self.y_true = np.arange(n_example, 0., -1.)
    self.y_pred_perfect = np.arange(n_example, 0., -1.)
    self.y_pred_random = np.random.permutation(self.y_pred_perfect)

  def test_gini(self):
    total_value = np.sum(self.y_true)
    cumulative_true = np.cumsum(self.y_true) / total_value
    gain_perfect = metrics.cumulative_true(
        self.y_true, self.y_pred_perfect)
    gain_random = metrics.cumulative_true(
        self.y_true, self.y_pred_random)
    gain = pd.DataFrame({
        'ground_truth': cumulative_true,
        'perfect_model': gain_perfect,
        'random_model': gain_random
    })
    gini = metrics.gini_from_gain(gain)
    self.assertEqual(gini.loc['perfect_model', 'normalized'], 1.)
    self.assertAlmostEqual(gini.loc['random_model', 'normalized'], 0., places=1)

  def test_decile_stats(self):
    decile_stats_perfect = metrics.decile_stats(self.y_true,
                                                self.y_pred_perfect)
    decile_stats_random = metrics.decile_stats(self.y_true,
                                               self.y_pred_random)
    self.assertTrue(np.all(decile_stats_perfect['normalized_rmse'] == 0))
    self.assertTrue(np.all(decile_stats_perfect['normalized_mae'] == 0))
    self.assertTrue(np.all(decile_stats_perfect['decile_mape'] == 0))
    self.assertTrue(
        np.allclose(
            decile_stats_random['label_mean'],
            np.random.permutation(decile_stats_random['label_mean']),
            rtol=1e-2,
            atol=1000))

  def test_gini_negative(self):
    test_df2 = """a,0.1,0.115
b,0.1,0.112
c,0.1,0.1151
d,0.9,0.01"""
    df = pd.read_csv(io.StringIO(test_df2), header=None)
    column_names = ['uid', 'label1', 'pred_scores']
    df.columns = column_names
    df = df.sort_values(by='label1', ascending=False)
    print(f"test_df: {df.head(10)}")


    total_value = np.sum(df['label1'])
    cumulative_true = np.cumsum(df['label1']) / total_value
    gain_model = metrics.cumulative_true(
        df['label1'], df['pred_scores'])
    gain = pd.DataFrame({
        'ground_truth': cumulative_true,
        'random_model': gain_model
    })
    gini = metrics.gini_from_gain(gain)
    print(f"test_gini: {gini.head(10)}")
    assert gini.loc['random_model', 'raw'] < 0


if __name__ == '__main__':
  unittest.main()
