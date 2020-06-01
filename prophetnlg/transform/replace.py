from collections import defaultdict
from enum import Enum
import itertools
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence, Union
from prophetnlg import Sentence, SentenceToken
from .base import SentenceTransform


class FillPolicy(str, Enum):
    IGNORE = 'ignore'
    ERROR = 'error'
    CYCLE = 'cycle'


class TokenFilter:
    def match(self, token: SentenceToken) -> bool:
        return True


class LemmaReplaceStreamTransform(SentenceTransform):
    def __init__(
        self,
        generator,
        replacement_stream: Iterable[SentenceToken],
        replace_pos: Iterable[str] = ('A', 'N', 'V'),
        token_filter: TokenFilter = TokenFilter(),
        fill_policy: FillPolicy = FillPolicy.IGNORE
    ):
        self.generator = generator
        self.token_filter = token_filter
        self.replace_pos = replace_pos
        n = len(replace_pos)
        pos_streams = itertools.tee(replacement_stream, n)
        self._replacement_streams = {
            pos: filter(lambda x, pos=pos: x.pos == pos, stream)
            for pos, stream in zip(replace_pos, pos_streams)
        }

    def _get_replacement(self, token: SentenceToken) -> SentenceToken:
        replacement = next(self._replacement_streams[token.pos])
        return self.generator.token_with_new_lemma(token, replacement)

    def _replace(self, token: SentenceToken) -> SentenceToken:
        if token.pos not in self.replace_pos:
            return token
        try:
            return self._get_replacement(token)
        except StopIteration:
            if self.fill_policy == FillPolicy.IGNORE:
                return token
            else:
                raise TypeError('Stream for category {category} drained!')

    def transform(self, sentence: Sentence) -> Sentence:
        return sentence._replace(
            tokens=[
                self._replace(t) for t in sentence.tokens
                if self.token_filter.match(t)
            ]
        )


class LemmaMapStreamTransform(LemmaReplaceStreamTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lemma_mappings = defaultdict(dict)

    def _get_replacement(self, token: SentenceToken) -> SentenceToken:
        pos = token.pos
        replacement = self._lemma_mappings[pos].get(token.root)
        if not replacement:
            replacement = next(self._replacement_streams[pos])
            self._lemma_mappings[pos][token.root] = replacement
        return self.generator.token_with_new_lemma(token, replacement)
