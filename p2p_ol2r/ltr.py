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

    def _get_result_pairs(self, query: str) -> list[str]:
        """
        Retrieve top-k results from semantic search and generate all possible combination pairs.

        Args:
            query: query string

        Returns:
            List of combination pairs of result IDs
        """
        results = [x for x, _ in self.embeddings.search(query, self.cfg.number_of_results)]
        return list(combinations(results, 2))

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
        
        pos_train_data = [ModelInput(
            query_vector,
            self.embeddings_map[results[selected_res]],
            self.embeddings_map[results[i]]
        ) for i in range(len(results)) if i != selected_res]

        return pos_train_data
    
    def result_ids_to_titles(self, results: list[str]) -> list[str]:
        return [self.metadata[x] for x in results]
    
    def query(self, query: str) -> dict[str, str]:
        """
        Determine ranking of results based on pairwise comparisons on the model.

        Args:
            query: query string
        
        Returns:
            Dict of result IDs to titles ordered by their ranking
        """
        query_vector = self.embed(query)
        results_combs = self._get_result_pairs(query)
        results_scores = {}

        self.model.eval()
        with torch.no_grad():
            for result_pair in results_combs:
                vec1 = self.embeddings_map[result_pair[0]]
                vec2 = self.embeddings_map[result_pair[1]]

                _, v = self.infer(ModelInput(query_vector, vec1, vec2))
                prob_1_over_2 = v[0] if type(v) == tuple else v
                prob_2_over_1 = v[1] if type(v) == tuple else 1-v

                
                k = result_pair[0]
                results_scores[k] = results_scores.get(k, 0) + prob_1_over_2
                k = result_pair[1]
                results_scores[k] = results_scores.get(k, 0) + prob_2_over_1

        # aggregating results by summing the probabilities of being superior
        results_scores = dict(sorted(results_scores.items(), key=itemgetter(1), reverse=True))
        ranked_results = {res_id: self.metadata[res_id] for res_id, _ in results_scores.items()}
        return ranked_results
    
    def on_result_selected(self, query: str, results: list[str], selected_res: int):
        """
        Retrains the model with the selected result as the most relevant.
        """
        self.train(self.gen_train_data(query, results, selected_res))
