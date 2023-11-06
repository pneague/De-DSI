import unittest
from p2p_ol2r.ltr import *
from tests import cfg

class TestGenTrainData(unittest.TestCase):

    ltr = LTR(cfg)

    def test_lengths(self):
        query = 'molecular tumor'
        results = [x for x, _ in self.ltr.embeddings.search(query, cfg.number_of_results)]
        train_data = self.ltr.gen_train_data(query, results, 0)
        self.assertEqual(len(train_data), cfg.number_of_results - 1)

if __name__ == "__main__":
    unittest.main()