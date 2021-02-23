from typing import Any, ClassVar, Iterable, List, Mapping, Optional, Sequence, Type
from pydantic import BaseModel
from prophetnlg import Sentence, SentenceToken
from prophetnlg.module import ConfigBase


class TransformBase:
    inputs: ClassVar[Mapping[str, Any]]
    outputs: ClassVar[Mapping[str, Any]]
    config_class: ClassVar[Type[ConfigBase]] = ConfigBase
    config: ConfigBase

    def __init__(self, config: Optional[ConfigBase] = None, **kwargs):
        if config:
            assert isinstance(config, self.config_class)
            assert not kwargs
            self.config = config
        else:
            self.config = self.config_class(**kwargs)
        self.original_config = self.config.copy()

    def reset(self):
        self.config = self.original_config


class SentenceTransformBase(TransformBase):
    inputs = {'sentence': Sentence}
    outputs = {'sentence': Sentence}

    def get_sentence(self, sentence: Sentence) -> Sentence:
        return sentence

    def transform(self, sentence: Sentence) -> Sentence:
        if sentence.passthrough:
            return sentence
        return self.get_sentence(sentence)

    def transform_sequence(self, sentences: Iterable[Sentence]) -> List[Sentence]:
        return [self.transform(s) for s in sentences]

    def transform_stream(self, sentences: Iterable[Sentence]) -> Iterable[Sentence]:
        for sentence in sentences:
            yield self.transform(sentence)


class SentenceToTextTransform(TransformBase):
    inputs = {'sentence': Sentence}
    outputs = {'text': str}

    def get_text(self, sentence: Sentence) -> str:
        return sentence.as_text()

    def transform(self, sentence: Sentence) -> str:
        if sentence.passthrough:
            return sentence.as_text()
        return self.get_text(sentence)

    def transform_sequence(self, sentences: Iterable[Sentence]) -> List[str]:
        return [self.transform(s) for s in sentences]

    def transform_stream(self, sentences: Iterable[Sentence]) -> Iterable[str]:
        for sentence in sentences:
            yield self.transform(sentence)
