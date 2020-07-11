import abc
from typing import Callable, Mapping, MutableMapping, Sequence, Union
from prophetnlg import Sentence, SentenceToken
from .base import SentenceTransformBase


class TokenAnnotationTransformBase(SentenceTransformBase):
    @abc.abstractmethod
    def annotate(self, token: SentenceToken) -> SentenceToken:
        pass

    def get_sentence(self, sentence: Sentence) -> Sentence:
        return sentence.replace(tokens=[self.annotate(t) for t in sentence.tokens])


class IncTokenPassThroughTransform(TokenAnnotationTransformBase):
    def passthrough_token(self, token: SentenceToken) -> int:
        return 1

    def annotate(self, token: SentenceToken) -> SentenceToken:
        return token.replace(passthrough=token.passthrough + self.passthrough_token(token))


class DecTokenPassThroughTransform(TokenAnnotationTransformBase):
    def annotate(self, token: SentenceToken) -> SentenceToken:
        return token.replace(passthrough=max(token.passthrough - 1, 0))
