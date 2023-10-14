import struct
import csv
import numpy as np
import torch
from txtai.embeddings import Embeddings
from itertools import combinations
from operator import itemgetter
from model import LTRModel

class LTR(LTRModel):
    """
    LTR class for learning to rank.
    
    Attributes:
        metadata: mapping of article uid to title
        embeddings_map: mapping of article uid to feature vector
        embeddings: txtai embeddings model
        results: local cache of query => results
    """
    metadata = {}
    embeddings_map = {}
    embeddings = None
    results = {}

    def __init__(self):
        super().__init__()

        with open('data/metadata.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.metadata[row[0]] = row[3]
        
        docs = []
        with open('data/embeddings.bin', 'rb') as embeddings_bin:
            format_str = '8s768f'
            while True:
                bin_data = embeddings_bin.read(struct.calcsize(format_str))
                if not bin_data:
                    break
                data = struct.unpack(format_str, bin_data)
                uid = data[0].decode('ascii').strip()
                features = list(data[1:])
                self.embeddings_map[uid] = features
                docs.append((uid, self.metadata[uid], features))

        self.embeddings = Embeddings({ 'path': 'allenai/specter' })
        self.embeddings.index(docs)

    def embed(self, x: str) -> list[float]:
        """
        Get vector representation of a (query) string.
        """
        return self.embeddings.batchtransform([(None, x, None)])[0]

    def gen_train_data(self, query: str, selected_res: int = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate training data. (We use sup(erior) and inf(erior) to denote relative relevance.)

        Args:
            query: query string
            selected_res: index of selected result
        
        Returns:
            tuple of (positive training data, negative training data)
        """
        results = self._query_model(query)
        query_vector = self.embed(query)
        pos_train_data, neg_train_data = [], []
        for i, _ in enumerate(results):
            # selected_res is superior to any other result
            if selected_res != None and i != selected_res:
                sup_doc = self.embeddings_map[results[selected_res]]
                inf_doc = self.embeddings_map[results[i]]
                pos_train_data.append(self.make_input(query_vector, sup_doc, inf_doc))
                neg_train_data.append(self.make_input(query_vector, inf_doc, sup_doc))
            
            # previous results are superior to current result
            for j in range(i):
                if j == selected_res: continue
                inf_doc = self.embeddings_map[results[i]]
                sup_doc = self.embeddings_map[results[j]]
                pos_train_data.append(self.make_input(query_vector, sup_doc, inf_doc))
                neg_train_data.append(self.make_input(query_vector, inf_doc, sup_doc))

            # current result is superior to next results
            for j in range(i+1, len(results)):
                if j == selected_res: continue
                sup_doc = self.embeddings_map[results[i]]
                inf_doc = self.embeddings_map[results[j]]
                pos_train_data.append(self.make_input(query_vector, sup_doc, inf_doc))
                neg_train_data.append(self.make_input(query_vector, inf_doc, sup_doc))

        pos_train_data = torch.from_numpy(np.array(pos_train_data))
        neg_train_data = torch.from_numpy(np.array(neg_train_data))
        return pos_train_data, neg_train_data
    
    def query(self, query: str) -> list[str]:
        """
        Returns ranked list of results (titles) for a query. 
        If results to this query are unknown, semantic search is performed, and the model is trained.
        """
        if query not in self.results: 
            # bootstrap model with semantic search results
            self.results[query] = [x for x, _ in self.embeddings.search(query, 5)]

        return [self.metadata[res] for res in self._query_model(query)]
    
    def _query_model(self, query: str):
        """
        Determine ranking of results in `self.results[query]` based on pairwise comparisons on the model.
        """
        query_vector = self.embed(query)
        results_combs = list(combinations(self.results[query], 2))
        results_scores = {}
        for result_pair in results_combs:
            vec1 = self.embeddings_map[result_pair[0]]
            vec2 = self.embeddings_map[result_pair[1]]
            is_sup = torch.round(
                    self.model(torch.from_numpy(self.make_input(query_vector, vec1, vec2)))
                ).item()
            k = result_pair[0]
            results_scores[k] = results_scores.get(k, 0) + is_sup
            k = result_pair[1]
            results_scores[k] = results_scores.get(k, 0) + (1 - is_sup)

        results_scores = dict(sorted(results_scores.items(), key=itemgetter(1), reverse=True))
        return [k for k, _ in results_scores.items()]
    
    def on_result_selected(self, query: str, selected_res: int):
        """
        Retrains the model with the selected result as the most relevant,
        and updates the local cache of results based on the updated model.
        """
        self.train(*self.gen_train_data(query, selected_res), 10)

        query_vector = self.embed(query)
        results_combs = list(combinations(self.results[query], 2))
        results_scores = {}
        for result_pair in results_combs:
            vec1 = self.embeddings_map[result_pair[0]]
            vec2 = self.embeddings_map[result_pair[1]]
            is_sup = torch.round(
                    self.model(torch.from_numpy(self.make_input(query_vector, vec1, vec2)))
                ).item()
            k = result_pair[0]
            results_scores[k] = results_scores.get(k, 0) + is_sup
            k = result_pair[1]
            results_scores[k] = results_scores.get(k, 0) + (1 - is_sup)

        results_scores = dict(sorted(results_scores.items(), key=itemgetter(1), reverse=True))
        self.results[query] = [k for k, _ in results_scores.items()]
