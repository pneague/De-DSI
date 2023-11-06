import struct
import csv
import numpy as np
import torch
from txtai.embeddings import Embeddings
from itertools import combinations
from operator import itemgetter
from .model import LTRModel, ModelInput
from .config import Config
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

class LTR(LTRModel):
    """
    LTR class for learning to rank.
    
    Attributes:
        metadata: mapping of article uid to title
        embeddings_map: mapping of article uid to feature vector
        embeddings: txtai embeddings model
    """
    metadata = {}
    embeddings_map = {}
    embeddings = None

    def __init__(self, cfg: Config):
        super().__init__(cfg)

        with open('data/metadata.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.metadata[row[0]] = row[3].strip()
        
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

        self.embeddings = Embeddings({ 'path': 'allenai/specter' })
        self.embeddings.load('data/embeddings_index.tar.gz')

    def _get_results(self, query: str) -> list[str]:
            """
            Returns an unordered list of relevant search results for the given query.

            Args:
                query: The search query.

            Returns:
                A list of document IDs
            """
            return [x for x, _ in self.embeddings.search(query, self.cfg.number_of_results)]

    def embed(self, x: str) -> list[float]:
        """
        Get vector representation of a (query) string.
        """
        return self.embeddings.batchtransform([(None, x, None)])[0]

    def gen_train_data(self, query: str, results: list[str], selected_res: int = None) -> list[ModelInput]:
        """
        Generate training data to be classified as _true_.

        Args:
            query: query string
            results: list of results (IDs)
            selected_res: index of selected result
        
        Returns:
            true training data
        """
        query_vector = self.embed(query)
        train_data = [ModelInput(
            query_vector,
            self.embeddings_map[results[selected_res]],
            self.embeddings_map[results[i]]
        ) for i in range(len(results)) if i != selected_res]
        return train_data
    
    def result_ids_to_titles(self, results: list[str]) -> list[str]:
        return [self.metadata[x] for x in results]
    
    def rank_results(self, query: np.ndarray, result_ids: list[str]) -> list[str]:
        """
        Determine ranking of results based on pairwise comparisons on the model.

        Args:
            query: The query to rank the results against.
            result_ids: The list of doc IDs to rank.

        Returns:
            The list of result IDs sorted by their relevance to the query.
        """
        result_pairs = list(combinations(result_ids, 2))
        result_scores = {id: 0 for id in result_ids}
        
        # aggregate inferred probabilities for each result pair
        for (d1_id, d2_id) in result_pairs:
            d1 = self.embeddings_map[d1_id]
            d2 = self.embeddings_map[d2_id]
            _, v = self.infer(ModelInput(query, d1, d2))
            prob_1_over_2 = v[0] if type(v) == tuple else v
            prob_2_over_1 = v[1] if type(v) == tuple else 1-v

            result_scores[d1_id] = result_scores.get(d1_id) + prob_1_over_2
            result_scores[d2_id] = result_scores.get(d2_id, 0) + prob_2_over_1

        # order result scores
        result_scores = dict(sorted(result_scores.items(), key=itemgetter(1), reverse=True))

        return [res_id for res_id, _ in result_scores.items()]
    
    def query(self, query: str) -> dict[str, str]:
        """
        Returns ranked results for the given query.

        Args:
            query: query string
        
        Returns:
            Dict of result IDs to titles ordered by their ranking
        """
        query_vector = self.embed(query)
        docs = self._get_results(query)
        ranked_results = self.rank_results(query_vector, docs)
        return {id: self.metadata[id] for id in ranked_results}
    
    def on_result_selected(self, query: str, results: list[str], selected_res: int):
        """
        Retrains the model with the selected result as the most relevant.
        """
        self.train(self.gen_train_data(query, results, selected_res))
