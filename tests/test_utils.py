import unittest
from p2p_ol2r.utils import *

class TestNDCG(unittest.TestCase):

    def test_perfect_score(self):
        self.assertEqual(
            ndcg(['a', 'b', 'c', 'd', 'e'], ['a', 'b', 'c', 'd', 'e']),
            1.0
        )

    def testmal_scores(self):
        self.assertGreater(
            ndcg(['a', 'd', 'c', 'b', 'e'], ['a', 'b', 'c', 'd', 'e']),
            ndcg(['e', 'd', 'c', 'b', 'a'], ['a', 'b', 'c', 'd', 'e']),
        )
        self.assertGreater(
            ndcg(['a', 'b', 'c', 'd', 'e'], ['a', 'b', 'c', 'd', 'e']),
            ndcg(['e', 'd', 'c', 'b', 'a'], ['a', 'b', 'c', 'd', 'e']),
        )

if __name__ == "__main__":
    unittest.main()