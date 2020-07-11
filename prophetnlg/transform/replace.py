import abc
import re
from collections import defaultdict
from enum import Enum
import itertools
from typing import Callable, Dict, Iterable, Iterator, Mapping, MutableMapping, Sequence, Union
from prophetnlg import Sentence, SentenceToken
from prophetnlg.generator.base import SentenceTokenGeneratorBase
from .base import SentenceTransformBase, ConfigBase


class FillPolicy(str, Enum):
    IGNORE = 'ignore'
    ERROR = 'error'
    CYCLE = 'cycle'


class LemmaStreamConfig(ConfigBase):
    generator: SentenceTokenGeneratorBase
    fill_policy: FillPolicy = FillPolicy.IGNORE
    replacements: Dict[str, Iterable[SentenceToken]]


class LemmaReplaceStreamConfig(LemmaStreamConfig):
    pass


class LemmaMapStreamConfig(LemmaStreamConfig):
    lemma_mappings: Mapping[str, Mapping[str, SentenceToken]] = defaultdict(dict)


class LemmaMapReplaceTransformBase(SentenceTransformBase):
    config_class = LemmaStreamConfig
    config: LemmaStreamConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.replacement_streams = {
            k: iter(v) for k, v in self.config.replacements.items()
        }

    @abc.abstractmethod
    def get_replacement(self, token: SentenceToken) -> SentenceToken:
        pass

    def _replace(self, token: SentenceToken) -> SentenceToken:
        if token.passthrough or not self.is_replaceable(token):
            return token
        try:
            return self.get_replacement(token)
        except StopIteration:
            if self.config.fill_policy == FillPolicy.IGNORE:
                return token
            else:
                raise TypeError('Stream for pos {token.pos} drained!')

    def is_replaceable(self, token: SentenceToken):
        return token.pos in self.replacement_streams

    def get_sentence(self, sentence: Sentence) -> Sentence:
        return sentence.replace(
            tokens=[self._replace(t) for t in sentence.tokens]
        )


class LemmaReplaceStreamTransform(LemmaMapReplaceTransformBase):
    config_class = LemmaReplaceStreamConfig
    config: LemmaReplaceStreamConfig

    def get_replacement(self, token: SentenceToken) -> SentenceToken:
        new_lemma = next(self.replacement_streams[token.pos])
        return self.config.generator.token_with_new_lemma(token, new_lemma)


class LemmaMapStreamTransform(LemmaMapReplaceTransformBase):
    config_class = LemmaMapStreamConfig
    config: LemmaMapStreamConfig

    def get_replacement(self, token: SentenceToken) -> SentenceToken:
        pos = token.pos
        replacement = self.config.lemma_mappings[pos].get(token.lemma)
        if not replacement:
            replacement = next(self.replacement_streams[pos])
            self.config.lemma_mappings[pos][token.lemma] = replacement
        return self.config.generator.token_with_new_lemma(token, replacement)
