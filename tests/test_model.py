import unittest
import numpy as np
import random
from p2p_ol2r.model import *
from p2p_ol2r.config import *
from tests import cfg

def setUp():
    ltr_model = LTRModel(cfg)
    mki = ltr_model.make_input # just an alias
    q = np.random.rand(768)
    docs = [np.random.rand(768) for _ in range(cfg.number_of_results)]
    return ltr_model, mki, cfg.number_of_results, q, docs

class TestModel(unittest.TestCase):

    def test_one_above_all(self):
        ltr_model, mki, k, q, docs = setUp()

        # train docs[0] to be above all others
        pos_train_data = np.array(
            [mki(q, docs[0], docs[i]) for i in range(1, k)]
        )
        neg_train_data = np.array(
            [mki(q, docs[i], docs[0]) for i in range(1, k)]
        )
        with silence(): 
            for _ in range(100):
                ltr_model.train(pos_train_data, neg_train_data)

        ltr_model.model.eval()
        with torch.no_grad():
            for i in range(1, k):
                res, _ = ltr_model.infer(q, docs[0], docs[i])
                self.assertTrue(res)
                res, _ = ltr_model.infer(q, docs[i], docs[0])
                self.assertFalse(res)
        
    def test_recent_bias(self):
        ltr_model, mki, k, q, docs = setUp()

        # We run 100 epochs on docs[0] first and then 20 on docs[1].
        # We expect that docs[0] will still be ranked higher than docs[1]
        # as there is this natural bias towards more recent training data.

        with silence(): 
            for _ in range(100):
                ltr_model.train(
                    np.array(
                        [mki(q, docs[0], docs[i]) for i in range(k) if i != 1]
                    ), 
                    np.array(
                        [mki(q, docs[i], docs[0]) for i in range(k) if i != 1]
                    )
                )
            
            for _ in range(20):
                ltr_model.train(
                    np.array(
                        [mki(q, docs[1], docs[i]) for i in range(k) if i != 1]
                    ), 
                    np.array(
                        [mki(q, docs[i], docs[1]) for i in range(k) if i != 1]
                    )
                )

        ltr_model.model.eval()
        with torch.no_grad():
            res, _ = ltr_model.infer(q, docs[0], docs[1])
            self.assertTrue(res)
            res, _ = ltr_model.infer(q, docs[1], docs[0])
            self.assertFalse(res)
    
    def test_how_much_it_takes(self):
        ltr_model, mki, _, q, docs = setUp()
        train_data = []

        train_data.extend([(
            np.array([mki(q, docs[0], docs[1])]), 
            np.array([mki(q, docs[1], docs[0])])
        )] * 100)

        train_data.extend([(
            np.array([mki(q, docs[1], docs[0])]),
            np.array([mki(q, docs[0], docs[1])])
        )] * 90)

        random.shuffle(train_data)

        for (pos_train_data, neg_train_data) in train_data:
            with silence(): ltr_model.train(pos_train_data, neg_train_data)
        
        with torch.no_grad():
            res, _ = ltr_model.infer(q, docs[0], docs[1])
            self.assertTrue(res)
            res, _ = ltr_model.infer(q, docs[1], docs[0])
            self.assertFalse(res)
        

    @unittest.skip("the ultimate test - doesn't work yet 🥲")
    def test_full_ranking(self):
        ltr_model, mki, k, q, docs = setUp()

        train_data = []

        for i in range(k-1):
            # docs[i] to be above all others
            pos_train_data = np.array(
                [mki(q, docs[i], docs[j]) for j in range(k) if i != j]
            )
            neg_train_data = np.array(
                [mki(q, docs[j], docs[i]) for j in range(k) if i != j]
            )
            epochs = max(0, k*10 - i*10)
            train_data.extend([(pos_train_data, neg_train_data)] * epochs)
        
        random.shuffle(train_data)

        for (pos_train_data, neg_train_data) in train_data:
            with silence(): ltr_model.train(pos_train_data, neg_train_data)

        ltr_model.model.eval()
        with torch.no_grad():    
            # expected: docs[0] > docs[1] > ... > docs[k-1]
            for i in range(k-1):
                print(f"docs[{i}] > docs[{i+1}]")
                res, values = ltr_model.infer(q, docs[i], docs[i+1])
                print(values)
                self.assertTrue(res)

                res, values = ltr_model.infer(q, docs[i+1], docs[i])
                print(values)
                self.assertFalse(res)

            # for i in range(k-1):
            #     for j in range(i+1, k):
            #         # self.assertGreater(
            #         #     ltr_model.model(torch.from_numpy(mki(q, docs[j], docs[i]))).item(),
            #         #     0.5
            #         # )
            #         self.assertLess(
            #             ltr_model.model(torch.from_numpy(mki(q, docs[j], docs[i]))).item(),
            #             0.5
            #         )
            #         break

if __name__ == "__main__":
    unittest.main()