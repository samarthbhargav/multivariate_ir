import logging

import faiss
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BaseFaissIPRetriever:
    def __init__(self, init_reps: np.ndarray):
        index = faiss.IndexFlatIP(init_reps.shape[1])
        self.index = index

    def add(self, p_reps: np.ndarray):
        try:
            self.index.add(p_reps)
        except TypeError:
            self.index.add(p_reps.astype(np.float32))

    def search(self, q_reps: np.ndarray, k: int):
        try:
            return self.index.search(q_reps, k)
        except TypeError:
            return self.index.search(q_reps.astype(np.float32), k)

    def batch_search(self, q_reps: np.ndarray, k: int, batch_size: int, quiet: bool = False):
        num_query = q_reps.shape[0]
        all_scores = []
        all_indices = []
        for start_idx in tqdm(range(0, num_query, batch_size), disable=quiet):
            nn_scores, nn_indices = self.search(q_reps[start_idx: start_idx + batch_size], k)
            all_scores.append(nn_scores)
            all_indices.append(nn_indices)
        all_scores = np.concatenate(all_scores, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)

        return all_scores, all_indices


class FaissRetriever(BaseFaissIPRetriever):
    def __init__(self, init_reps: np.ndarray, factory_str: str):
        index = faiss.index_factory(init_reps.shape[1], factory_str)
        self.index = index
        self.index.verbose = True
        if not self.index.is_trained:
            self.index.train(init_reps)
