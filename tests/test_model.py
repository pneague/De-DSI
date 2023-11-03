import unittest
import numpy as np
import random
from p2p_ol2r.model import *

def setUp(k = 10):
    ltr_model = LTRModel(False)
    mki = ltr_model.make_input # just an alias
    q = np.random.rand(768)
    docs = [np.random.rand(768) for _ in range(k)]
    return ltr_model, mki, k, q, docs
class TestModel(unittest.TestCase):

    def test_one_above_all(self):
        ltr_model, mki, k, q, docs = setUp()

        # train docs[0] to be above all others
        pos_train_data = torch.from_numpy(np.array(
            [mki(q, docs[0], docs[i]) for i in range(1, k)]
        ))
        neg_train_data = torch.from_numpy(np.array(
            [mki(q, docs[i], docs[0]) for i in range(1, k)]
        ))
        with silence(): ltr_model.train(pos_train_data, neg_train_data, 100)

        ltr_model.model.eval()
        torch.no_grad()

        for i in range(1, k):
            self.assertGreater(
                ltr_model.model(torch.from_numpy(mki(q, docs[0], docs[i]))).item(),
                0.5
            )
            self.assertLess(
                ltr_model.model(torch.from_numpy(mki(q, docs[i], docs[0]))).item(),
                0.5
            )
    
    def test_recent_bias(self):
        ltr_model, mki, k, q, docs = setUp()

        # We run 100 epochs on docs[0] first and then 20 on docs[1].
        # We expect that docs[0] will still be ranked higher than docs[1]
        # as there is this natural bias towards more recent training data.

        with silence(): 
            ltr_model.train(
                torch.from_numpy(np.array(
                    [mki(q, docs[0], docs[i]) for i in range(1, k)]
                )), 
                torch.from_numpy(np.array(
                    [mki(q, docs[i], docs[0]) for i in range(1, k)]
                )), 
                100
            )

            ltr_model.train(
                torch.from_numpy(np.array(
                    [mki(q, docs[1], docs[i]) for i in range(k) if i != 1]
                )), 
                 torch.from_numpy(np.array(
                    [mki(q, docs[i], docs[1]) for i in range(k) if i != 1]
                )), 
                20
            )

        ltr_model.model.eval()
        torch.no_grad()
        self.assertGreater(
            ltr_model.model(torch.from_numpy(mki(q, docs[0], docs[1]))).item(),
            0.5
        )
        self.assertLess(
            ltr_model.model(torch.from_numpy(mki(q, docs[1], docs[0]))).item(),
            0.5
        )
        

    @unittest.skip("the ultimate test - doesn't work yet 🥲")
    def test_full_ranking(self):
        ltr_model, mki, k, q, docs = setUp()

        train_data = []

        for i in range(k-1):
            # docs[i] to be above all others
            pos_train_data = torch.from_numpy(np.array(
                [mki(q, docs[i], docs[j]) for j in range(k) if i != j]
            ))
            neg_train_data = torch.from_numpy(np.array(
                [mki(q, docs[j], docs[i]) for j in range(k) if i != j]
            ))
            train_data.extend([(pos_train_data, neg_train_data)] * (k*100 - i*100))
        
        random.shuffle(train_data)

        for (pos_train_data, neg_train_data) in train_data:
            with silence(): ltr_model.train(pos_train_data, neg_train_data, 1)

        ltr_model.model.eval()
        torch.no_grad()
        
        for i in range(k-1):
            for j in range(i+1, k):
                self.assertGreater(
                    ltr_model.model(torch.from_numpy(mki(q, docs[i], docs[j]))).item(),
                    0.5
                )
                # self.assertLess(
                #     ltr_model.model(torch.from_numpy(mki(q, docs[j], docs[i]))).item(),
                #     0.5
                # )

if __name__ == "__main__":
    unittest.main()