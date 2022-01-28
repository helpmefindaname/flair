from typing import List

import torch
from tqdm import tqdm

from flair.data import Sentence
from flair.embeddings import FlairEmbeddings


class TrueCaser:
    def __init__(self, forward_embedding: FlairEmbeddings, backward_embedding: FlairEmbeddings):
        self.forward_embedding = forward_embedding
        self.backward_embedding = backward_embedding
        super().__init__()

    def transform_sentence(self, sentence: Sentence):
        titled_text = sentence.to_tokenized_string().title()
        start_marker = "\n"
        end_marker = " "
        forward_outputs = self.forward_embedding.lm.get_predictions([titled_text], start_marker, end_marker)[:, 0, :]
        backward_outputs = self.backward_embedding.lm.get_predictions([titled_text], start_marker, end_marker)[:, 0, :]
        backward_outputs = backward_outputs.flip(0)
        offset = len(start_marker)
        for token in sentence:
            lower_c = token.text[0].lower()
            upper_c = lower_c.upper()
            l_fw_idx, u_fw_idx = self.forward_embedding.lm.dictionary.get_idx_for_items([lower_c, upper_c])
            l_bw_idx, u_bw_idx = self.backward_embedding.lm.dictionary.get_idx_for_items([lower_c, upper_c])
            do_upper = (
                    backward_outputs[offset, u_bw_idx] + forward_outputs[offset, u_fw_idx]
                    > backward_outputs[offset, l_bw_idx] + forward_outputs[offset, l_fw_idx]
            )

            if do_upper:
                token.text = token.text.title()
            else:
                token.text = token.text.lower()

            offset += len(token.text) + 1

    def transform_sentences(self, sentences: List[Sentence]):
        for sent in tqdm(sentences, desc="true casing sentences"):
            self.transform_sentence(sent)
