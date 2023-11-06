import unittest
from p2p_ol2r.ltr import *
from tests import cfg

class TestLTR(unittest.TestCase):

    ltr = LTR(cfg)

    def test_gen_train_data(self):
        query = 'molecular tumor'
        results = [x for x, _ in self.ltr.embeddings.search(query, cfg.number_of_results)]
        train_data = self.ltr.gen_train_data(query, results, 0)
        self.assertEqual(len(train_data), cfg.number_of_results - 1)
    
    def test_rank_results(self):
        q = np.random.rand(768)
        k = cfg.number_of_results
        doc_ids = [f'id{i}abcd' for i in range(k)]
        self.ltr.embeddings_map = {id: np.random.rand(768) for id in doc_ids}

        # We mock `infer` to respond positively for each query.
        # We expect the results to be ordered as they are in `doc_ids`.
        for r in [
            (True, (1.0, 0.0)), (True, (0.51, 0.49)),
            (True, 1.0), (True, 0.51)
        ]:
            with unittest.mock.patch('p2p_ol2r.model.LTRModel.infer', new=lambda _, __: r):
                ordered_docs = self.ltr.rank_results(q, doc_ids)
                self.assertListEqual(ordered_docs, doc_ids)
    
        # We mock `infer` to respond negatively for each query.
        # We expect the results to be ordered reversed to what they are in `doc_ids`.
        for r in [
            (False, (0.0, 1.0)), (False, (0.49, 0.51)),
            (False, 0.0), (False, 0.49)
        ]:
            with unittest.mock.patch('p2p_ol2r.model.LTRModel.infer', new=lambda _, __: r):
                ordered_docs = self.ltr.rank_results(q, doc_ids)
                self.assertListEqual(ordered_docs, list(reversed(doc_ids)))

if __name__ == "__main__":
    unittest.main()