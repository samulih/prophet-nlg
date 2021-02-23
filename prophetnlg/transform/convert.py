from operator import attrgetter
import itertools
from typing import Dict, Iterable, Iterator, List
from prophetnlg import Sentence, SentenceToken
from .base import ConfigBase, TransformBase


class TokenCategoryDemultiplexerConfig(ConfigBase):
    categories: List[str]
    category_attr: str = 'pos'


class SentencesToTokensTransform(TransformBase):
    def transform(self, sentence: Sentence) -> List[SentenceToken]:
        if sentence.passthrough:
            return []
        return [t for t in sentence.tokens if not t.passthrough]

    def transform_sequence(self, sentences: Iterable[Sentence]) -> List[SentenceToken]:
        return list(self.transform_stream(sentences))

    def transform_stream(self, sentences: Iterable[Sentence]) -> Iterator[SentenceToken]:
        for sentence in sentences:
            yield from self.transform(sentence)


class TokenCategoryDemultiplexerTransform(TransformBase):
    config_class = TokenCategoryDemultiplexerConfig
    config: TokenCategoryDemultiplexerConfig

    def transform_stream(self, tokens: Iterable[SentenceToken]) -> Dict[str, Iterator[SentenceToken]]:
        streams = itertools.tee(tokens, len(self.config.categories))
        get_category = attrgetter(self.config.category_attr)

        return {
            category: filter((lambda token, c=category: get_category(token) == c), stream)
            for category, stream in zip(self.config.categories, streams)
        }
