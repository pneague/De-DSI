import unittest
from unittest.mock import patch
from p2p_ol2r.ltr import *
from tests import cfg

class TestLTR(unittest.TestCase):

    ltr = LTR(cfg)

    @patch('p2p_ol2r.ltr.LTR.embed', new=lambda _, x: np.array([0.123] * 768))
    @patch.dict(LTR.embeddings_map, {
        'id0': np.array([.0] * 768),
        'id1': np.array([.1] * 768),
        'id2': np.array([.2] * 768),
        'id3': np.array([.3] * 768),
    })
    def test_gen_train_data(self):
        query = 'molecular tumor'
        query_vec = np.array([.123] * 768)
        results = [f'id{i}' for i in range(4)]
        train_data = self.ltr.gen_train_data(query, results, 1)
        expected_train_data = [
            ModelInput(query_vec, np.array([.1] * 768), np.array([.0] * 768)),
            ModelInput(query_vec, np.array([.1] * 768), np.array([.2] * 768)),
            ModelInput(query_vec, np.array([.1] * 768), np.array([.3] * 768))
        ]
        self.assertEqual(len(train_data), len(expected_train_data))
        for generated, expected in zip(train_data, expected_train_data):
            self.assertTrue(torch.allclose(generated, expected))
    
    def test_rank_results(self):
        q = np.random.rand(768)
        k = cfg.number_of_results
        doc_ids = [f'id{i}' for i in range(k)]
        self.ltr.embeddings_map = {id: np.random.rand(768) for id in doc_ids}

        # We mock `infer` to respond positively for each query.
        # We expect the results to be ordered as they are in `doc_ids`.
        for r in [(True, (1.0, 0.0)), (True, (0.51, 0.49)), (True, 1.0), (True, 0.51)]:
            with patch('p2p_ol2r.model.LTRModel.infer', lambda _, __: r):
                ordered_docs = self.ltr.rank_results(q, doc_ids)
                self.assertListEqual(ordered_docs, doc_ids)
    
        # We mock `infer` to respond negatively for each query.
        # We expect the results to be ordered reversed to what they are in `doc_ids`.
        for r in [(False, (0.0, 1.0)), (False, (0.49, 0.51)), (False, 0.0), (False, 0.49)]:
            with patch('p2p_ol2r.model.LTRModel.infer', lambda _, __: r):
                ordered_docs = self.ltr.rank_results(q, doc_ids)
                self.assertListEqual(ordered_docs, list(reversed(doc_ids)))

if __name__ == "__main__":
    unittest.main()