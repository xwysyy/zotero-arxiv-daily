from abc import ABC, abstractmethod
from omegaconf import DictConfig
from ..protocol import Paper, CorpusPaper
import numpy as np
from typing import Type
class BaseReranker(ABC):
    def __init__(self, config:DictConfig):
        self.config = config

    def rerank(self, candidates:list[Paper], corpus:list[CorpusPaper]) -> list[Paper]:
        if len(corpus) == 0:
            return candidates
        corpus = sorted(corpus,key=lambda x: x.added_date,reverse=True)
        time_decay_weight = np.exp(-0.05 * np.arange(len(corpus)))
        collection_weight = np.array([0.3 if any(p.startswith('old/') for p in c.paths) else 1.0 for c in corpus])
        weight: np.ndarray = time_decay_weight * collection_weight
        weight_sum = weight.sum()
        if weight_sum <= 0:
            return candidates
        weight = weight / weight_sum
        candidate_text = [c.title + '. ' + c.abstract for c in candidates]
        corpus_text = [c.title + '. ' + c.abstract for c in corpus]
        sim = self.get_similarity_score(candidate_text, corpus_text)
        assert sim.shape == (len(candidates), len(corpus))
        scores = (sim * weight).sum(axis=1) * 10 # [n_candidate]
        for s,c in zip(scores,candidates):
            c.score = float(s)
        candidates = sorted(candidates,key=lambda x: x.score,reverse=True)
        return candidates
    
    @abstractmethod
    def get_similarity_score(self, s1:list[str], s2:list[str]) -> np.ndarray:
        raise NotImplementedError

registered_rerankers = {}

def register_reranker(name:str):
    def decorator(cls):
        registered_rerankers[name] = cls
        return cls
    return decorator

def get_reranker_cls(name:str) -> Type[BaseReranker]:
    if name not in registered_rerankers:
        raise ValueError(f"Reranker {name} not found")
    return registered_rerankers[name]
