
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import pandapower as pp
import os
import sys

# Ensure poc is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from poc.pipeline import run_pipeline

class TestRetrySemantics(unittest.TestCase):
    def setUp(self):
        # Create a tiny network
        self.net = pp.create_empty_network()
        pp.create_bus(self.net, vn_kv=110)
        pp.create_gen(self.net, bus=0, p_mw=100, vm_pu=1.0, max_p_mw=200, min_p_mw=0)
        # Mock converged attribute
        self.net.converged = True

    @patch('poc.pipeline.get_violations')
    @patch('poc.pipeline.compute_sensitivity')
    @patch('poc.pipeline.llm_intent')
    @patch('poc.pipeline.intent_to_action')
    @patch('poc.pipeline.check_0')
    @patch('poc.pipeline.check_1')
    @patch('poc.metrics.evaluate_action_cost')
    def test_semantic_a_retries(self, mock_cost, mock_check1, mock_check0, mock_intent_to_action, mock_llm_intent, mock_sens, mock_get_viols):
        # Setup initial violations
        initial_viols = {
            'total_count': 1, 
            'voltage_low': [(0, 0.9, 4.2)],
            'voltage_high': [],
            'thermal': []
        }
        mock_get_viols.return_value = initial_viols
        mock_sens.return_value = {'gen_indices': [0], 'dV_dVset': np.zeros((1, 1)), 'dLoading_dPg': np.zeros((0, 1))}
        mock_cost.return_value = {'cost_dp': 0, 'cost_dv': 0, 'cost_total': 0, 'n_act': 0}
        
        # First attempt: Fail Check-1
        mock_llm_intent.side_effect = [
            ({'generators': [0], 'targets': [], 'source': 'llm'}, 100, "raw1"), # First attempt
            ({'generators': [0], 'targets': [], 'source': 'llm'}, 100, "raw2")  # Second attempt
        ]
        mock_intent_to_action.return_value = {0: {'dp': 10, 'dv': 0.05}}
        mock_check0.return_value = (True, [])
        
        # Residual violations for first failure
        residual_viols = {
            'total_count': 1, 
            'voltage_low': [(0, 0.92, 2.1)],
            'voltage_high': [],
            'thermal': []
        }
        mock_check1.side_effect = [
            (False, residual_viols, self.net), # First call fails
            (True, {'total_count': 0}, self.net) # Second call succeeds
        ]
        
        # Run pipeline
        res = run_pipeline(self.net, mode='llm', api_key='fake', verbose=True)
        
        # Verify llm_intent calls
        self.assertEqual(mock_llm_intent.call_count, 2)
        
        # Call 1: violations should be initial_viols, hint should be empty (fixed by base_hint)
        args1, kwargs1 = mock_llm_intent.call_args_list[0]
        self.assertEqual(args1[0], initial_viols)
        self.assertIsNone(kwargs1['hint'])
        
        # Call 2: violations should STILL be initial_viols (Semantic A), hint should contain failure info
        args2, kwargs2 = mock_llm_intent.call_args_list[1]
        self.assertEqual(args2[0], initial_viols) 
        self.assertIn("Check-1 (Steady-State AC Flow) FAILED", kwargs2['hint'])
        self.assertIn("1 violations STILL REMAIN", kwargs2['hint'])
        
        self.assertTrue(res['success'])
        self.assertEqual(res['n_llm_calls'], 2)

if __name__ == '__main__':
    unittest.main()
