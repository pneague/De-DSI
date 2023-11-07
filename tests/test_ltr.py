import unittest
from unittest.mock import patch
from p2p_ol2r.ltr import *
from p2p_ol2r.utils import *
from tests import cfg

@patch('p2p_ol2r.ltr.LTR.embed', new=lambda _, __: np.array([0.123] * 768)) # embeds query
@patch('p2p_ol2r.ltr.LTR._get_results', new=lambda _, __: [f'id{i}' for i in range(cfg.number_of_results)])
class TestLTR(unittest.TestCase):

    def setUp(self) -> None:
        self.ltr = LTR(cfg)
        self.ltr.embeddings_map = {
            f'id{i}': np.random.RandomState(i).rand(768) for i in range(cfg.number_of_results)
        }
        self.ltr.metadata = {
            f'id{i}': f'title{i}' for i in range(cfg.number_of_results)
        }
        self.doc_ids = [f'id{i}' for i in range(cfg.number_of_results)]

    def test_gen_train_data(self):
        query = 'molecular tumor'
        query_vec = np.array([.123] * 768)
        selected_res = 1
        train_data = self.ltr.gen_train_data(query, self.doc_ids, selected_res)
        expected_train_data = [
            ModelInput(
                query_vec, 
                np.random.RandomState(selected_res).rand(768), np.random.RandomState(i).rand(768)
            ) for i in range(cfg.number_of_results) if i != selected_res
        ]
        self.assertEqual(len(train_data), len(expected_train_data))
        for generated, expected in zip(train_data, expected_train_data):
            self.assertTrue(torch.allclose(generated, expected))
    
    def test_rank_results(self):
        q = np.random.rand(768)

        # We mock `infer` to respond positively for each query.
        # We expect the results to be ordered as they are in `doc_ids`.
        for r in [(True, (1.0, 0.0)), (True, (0.51, 0.49)), (True, 1.0), (True, 0.51)]:
            with patch('p2p_ol2r.model.LTRModel.infer', lambda _, __: r):
                ordered_docs = self.ltr.rank_results(q, self.doc_ids)
                self.assertListEqual(ordered_docs, self.doc_ids)
    
        # We mock `infer` to respond negatively for each query.
        # We expect the results to be ordered reversed to what they are in `doc_ids`.
        for r in [(False, (0.0, 1.0)), (False, (0.49, 0.51)), (False, 0.0), (False, 0.49)]:
            with patch('p2p_ol2r.model.LTRModel.infer', lambda _, __: r):
                ordered_docs = self.ltr.rank_results(q, self.doc_ids)
                self.assertListEqual(ordered_docs, list(reversed(self.doc_ids)))

    def test_query(self):
        # We mock `infer` to respond positively for each query.
        # We expect the results to be ordered as they are in `doc_ids`.
        for r in [(True, (1.0, 0.0)), (True, (0.51, 0.49)), (True, 1.0), (True, 0.51)]:
            with patch('p2p_ol2r.model.LTRModel.infer', lambda _, __: r):
                self.assertDictEqual(self.ltr.query('molecular tumor'), {
                    f'id{i}': f'title{i}' for i in range(cfg.number_of_results)
                })
        # We mock `infer` to respond negatively for each query.
        # We expect the results to be ordered reversed to what they are in `doc_ids`.
        for r in [(False, (0.0, 1.0)), (False, (0.49, 0.51)), (False, 0.0), (False, 0.49)]:
            with patch('p2p_ol2r.model.LTRModel.infer', lambda _, __: r):
                self.assertDictEqual(self.ltr.query('molecular tumor'), {
                    f'id{i}': f'title{i}' for i in reversed(range(cfg.number_of_results))
                })

    def test_train_and_query(self):
        k = cfg.number_of_results
        q = self.ltr.embed('molecular tumor')
        docs = list(self.ltr.embeddings_map.values())
        train_data = []

        for i in range(k-1):
            # docs[i] to be above all others
            i_over_all = [ModelInput(q, docs[i], docs[j]) for j in range(k) if i != j]
            epochs = max(0, k*10 - i*10)
            train_data.extend(i_over_all * epochs)
        
        with silence(): self.ltr.train(train_data)

        res = self.ltr.query('molecular tumor')
        self.assertDictEqual(res, {
            f'id{i}': f'title{i}' for i in range(cfg.number_of_results)
        })

if __name__ == "__main__":
    unittest.main()