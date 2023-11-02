import unittest
import io
import random
from p2p_ol2r.utils import *

class TestSplit(unittest.TestCase):

    def test_clean_split(self):
        a = split(io.BytesIO(bytes(1024)), 512)
        self.assertEqual(len(a), 2)
        self.assertEqual(len(a[0]), 512)
        self.assertEqual(len(a[1]), 512)

    def test_dirty_split(self):
        a = split(io.BytesIO(bytes(1025)), 512)
        self.assertEqual(len(a), 3)
        self.assertEqual(len(a[0]), 512)
        self.assertEqual(len(a[1]), 512)
        self.assertEqual(len(a[2]), 1)

    def test_split_size_exceeds_input(self):
        a = split(io.BytesIO(bytes(123)), 1024)
        self.assertEqual(len(a), 1)
        self.assertEqual(len(a[0]), 123)

    def test_zero_input(self):
        a = split(io.BytesIO(bytes(0)), 512)
        self.assertEqual(len(a), 0)

class TestNDCG(unittest.TestCase):

    perfect_ranking = ['a', 'b', 'c', 'd', 'e']

    def test_perfect_score(self):
        self.assertEqual(
            ndcg(self.perfect_ranking, self.perfect_ranking),
            1.0
        )

    def test_mal_scores(self):
        self.assertGreater(
            ndcg(['a', 'd', 'c', 'b', 'e'], self.perfect_ranking),
            ndcg(['e', 'd', 'c', 'b', 'a'], self.perfect_ranking),
        )
        self.assertLess(
            ndcg(['e', 'd', 'c', 'b', 'a'], self.perfect_ranking),
            1.0
        )

if __name__ == "__main__":
    unittest.main()