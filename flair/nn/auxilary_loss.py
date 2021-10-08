from typing import List

import torch

from flair.data import Sentence


class AuxilaryLoss(torch.nn.Module):

    def forward_loss(self, sentence_features, sentences: List[Sentence]) -> float:
        raise NotImplementedError
