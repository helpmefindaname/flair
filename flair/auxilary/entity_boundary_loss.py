from typing import List

import torch
from torch import nn

import flair
from flair.data import Sentence, Dictionary
from flair.models.sequence_tagger_model import pad_tensors, log_sum_exp_batch, STOP_TAG, START_TAG


class EntityBoundaryLoss(torch.nn.Module):

    def __init__(self, hidden_size: int, scale: float = 1.0, tag_type: str = "ner"):
        self.tag_dictionary = Dictionary(add_unk=False)
        self.scale = scale
        self.tag_type = tag_type
        for c in "BIESO":
            self.tag_dictionary.add_item(c)
        self.tag_dictionary.add_item(START_TAG)
        self.tag_dictionary.add_item(STOP_TAG)
        self.tagset_size = len(self.tag_dictionary)

        super().__init__()
        self.linear = nn.Linear(hidden_size, self.tagset_size)
        self.transitions = torch.nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size)
        )

        self.transitions.detach()[
        self.tag_dictionary.get_idx_for_item(START_TAG), :
        ] = -10000

        self.transitions.detach()[
        :, self.tag_dictionary.get_idx_for_item(STOP_TAG)
        ] = -10000
        self.to(flair.device)

    def _tag_indexes(self, sentence: Sentence) -> List[int]:
        tags = "".join([token.get_tag(self.tag_type).value[0] for token in sentence.tokens])
        while "BB" in tags or "IB" in tags:
            tags = tags.replace("BB", "SB")
            tags = tags.replace("IB", "EB")
        tags = tags.replace("IO", "EO")
        tags = tags.replace("BO", "SO")

        return self.tag_dictionary.get_idx_for_items(list(tags))

    def forward_loss(self, sentence_features, sentences: List[Sentence]) -> float:

        features = self.linear(sentence_features)

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]

        tag_list: List = []
        token_count = 0
        for sentence in sentences:
            # get the tags in this sentence
            tag_idx: List[int] = self._tag_indexes(sentence)
            token_count += len(tag_idx)
            # add tags as tensor
            tag = torch.tensor(tag_idx, device=flair.device)
            tag_list.append(tag)

        tags, _ = pad_tensors(tag_list)

        forward_score = self._forward_alg(features, lengths)
        gold_score = self._score_sentence(features, tags, lengths)

        score = forward_score - gold_score

        return score.sum()

    def _forward_alg(self, feats, lens_):

        init_alphas = torch.FloatTensor(self.tagset_size).fill_(-10000.0)
        init_alphas[self.tag_dictionary.get_idx_for_item(START_TAG)] = 0.0

        forward_var = torch.zeros(
            feats.shape[0],
            feats.shape[1] + 1,
            feats.shape[2],
            dtype=torch.float,
            device=flair.device,
        )

        forward_var[:, 0, :] = init_alphas[None, :].repeat(feats.shape[0], 1)

        transitions = self.transitions.view(
            1, self.transitions.shape[0], self.transitions.shape[1]
        ).repeat(feats.shape[0], 1, 1)

        for i in range(feats.shape[1]):
            emit_score = feats[:, i, :]

            tag_var = (
                    emit_score[:, :, None].repeat(1, 1, transitions.shape[2])
                    + transitions
                    + forward_var[:, i, :][:, :, None]
                    .repeat(1, 1, transitions.shape[2])
                    .transpose(2, 1)
            )

            max_tag_var, _ = torch.max(tag_var, dim=2)

            tag_var = tag_var - max_tag_var[:, :, None].repeat(
                1, 1, transitions.shape[2]
            )

            agg_ = torch.log(torch.sum(torch.exp(tag_var), dim=2))

            cloned = forward_var.clone()
            cloned[:, i + 1, :] = max_tag_var + agg_

            forward_var = cloned

        forward_var = forward_var[range(forward_var.shape[0]), lens_, :]

        terminal_var = forward_var + self.transitions[
                                         self.tag_dictionary.get_idx_for_item(STOP_TAG)
                                     ][None, :].repeat(forward_var.shape[0], 1)

        alpha = log_sum_exp_batch(terminal_var)

        return alpha

    def _score_sentence(self, feats, tags, lens_):

        start = torch.tensor(
            [self.tag_dictionary.get_idx_for_item(START_TAG)], device=flair.device
        )
        start = start[None, :].repeat(tags.shape[0], 1)

        stop = torch.tensor(
            [self.tag_dictionary.get_idx_for_item(STOP_TAG)], device=flair.device
        )
        stop = stop[None, :].repeat(tags.shape[0], 1)

        pad_start_tags = torch.cat([start, tags], 1)
        pad_stop_tags = torch.cat([tags, stop], 1)

        for i in range(len(lens_)):
            pad_stop_tags[i, lens_[i]:] = self.tag_dictionary.get_idx_for_item(
                STOP_TAG
            )

        score = torch.FloatTensor(feats.shape[0]).to(flair.device)

        for i in range(feats.shape[0]):
            r = torch.LongTensor(range(lens_[i])).to(flair.device)

            score[i] = torch.sum(
                self.transitions[
                    pad_stop_tags[i, : lens_[i] + 1], pad_start_tags[i, : lens_[i] + 1]
                ]
            ) + torch.sum(feats[i, r, tags[i, : lens_[i]]])

        return score
