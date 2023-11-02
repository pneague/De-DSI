import unittest
import logging
from p2p_ol2r.ltr import *

class TestGenTrainData(unittest.TestCase):

    ltr = LTR(5, False)

    def test_lengths(self):
        query = 'molecular tumor'
        results = [x for x, _ in self.ltr.embeddings.search(query, self.ltr.number_of_results)]
        pos, neg = self.ltr.gen_train_data(query, results, 0)
        
        log = logging.getLogger( "TestGenTrainData" )
        log.debug(pos)

        self.assertEqual(len(pos), self.ltr.number_of_results - 1)
        self.assertEqual(len(pos), len(neg))

if __name__ == "__main__":
    unittest.main()