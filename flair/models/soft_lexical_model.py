from pathlib import Path
from typing import Union, List, Optional, Tuple

import numpy as np
import torch
from gensim.models import FastText
from gensim.models.keyedvectors import Word2VecKeyedVectors
from torch.utils.data import Dataset

from flair.data import Sentence, Corpus, _iter_dataset, Dictionary
from flair.nn import Model
from flair.training_utils import Result


class SoftLexicalModel(Model[Sentence]):
    def __init__(self, label_type: str, tag_dict: Dictionary, hidden=100):
        self._label_type = label_type
        # double the window size, as it represents double the tokens
        self.word2vec = FastText(vector_size=hidden, window=10, min_count=2)
        self.tag_dict = tag_dict
        super().__init__()

    def train_model(
        self, corpus: Corpus, save_path: Path, train_with_dev: bool = True, epochs: int = 5, lr: float = 0.025
    ):

        word2vec_corpus = list(map(Sentence.to_full_tagged_tokens, _iter_dataset(corpus.train)))
        if train_with_dev:
            word2vec_corpus.extend(map(Sentence.to_full_tagged_tokens, _iter_dataset(corpus.dev)))

        self.word2vec.build_vocab(word2vec_corpus)
        self.word2vec.train(word2vec_corpus, total_examples=len(word2vec_corpus), epochs=epochs, start_alpha=lr)
        self.word2vec.wv.adjust_vectors()

        tag_names_set = set(self.tag_dict.get_items())
        non_tag_words = [word for word in self.word2vec.wv.index_to_key if word not in tag_names_set]
        tag_vector_ids = np.array(
            [self.word2vec.wv.key_to_index[tag] for tag in tag_names_set if tag in self.word2vec.wv.key_to_index]
        )

        word_vector_ids = np.array([self.word2vec.wv.key_to_index[word] for word in non_tag_words])
        vectors = self.word2vec.wv.vectors
        vectors /= np.linalg.norm(vectors, ord=2, axis=1, keepdims=True)
        word_vectors = vectors[word_vector_ids]
        tag_vectors = vectors[tag_vector_ids]
        actual_word_vectors = word_vectors @ tag_vectors.T

        kv = Word2VecKeyedVectors(len(tag_vector_ids))
        kv.vectors = actual_word_vectors
        kv.key_to_index = {word: i for i, word in enumerate(non_tag_words)}
        kv.save(str(save_path))

    @property
    def label_type(self):
        return self._label_type

    def forward_loss(
        self, data_points: Union[List[Sentence], Sentence]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
        pass

    def evaluate(
        self,
        data_points: Union[List[Sentence], Dataset],
        gold_label_type: str,
        out_path: Union[str, Path] = None,
        embedding_storage_mode: str = "none",
        mini_batch_size: int = 32,
        num_workers: Optional[int] = 8,
        **kwargs,
    ) -> Result:
        pass

    def _get_state_dict(self):
        pass

    @staticmethod
    def _init_model_with_state_dict(state):
        pass
