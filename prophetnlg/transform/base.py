from typing import Any, ClassVar, Generic, Iterable, Mapping, Optional, Type, TypeVar, Union
from pydantic import BaseModel
from prophetnlg import Sentence, SentenceToken


T = TypeVar('T')


class ConfigBase(BaseModel, Generic[T]):
    def reset(self) -> None:
        pass


class TransformBase:
    inputs: ClassVar[Mapping[str, Any]]
    outputs: ClassVar[Mapping[str, Any]]
    config_class: ClassVar[Type[ConfigBase]] = ConfigBase
    config: ConfigBase

    def __init__(self, config: Optional[ConfigBase] = None, **kwargs):
        if config:
            assert isinstance(config, self.config_class)
            self.config = config
        else:
            self.config = self.config_class(**kwargs)

    def reset(self):
        self.config.reset()


class SentenceTransformBase(TransformBase):
    inputs = {'sentence': Sentence}
    outputs = {'sentence': Sentence}

    def get_sentence(self, sentence: Sentence) -> Sentence:
        return sentence

    def transform(self, sentence: Sentence) -> Sentence:
        if sentence.passthrough:
            return sentence
        return self.get_sentence(sentence)

    def transform_stream(self, sentences: Iterable[Sentence]) -> Iterable[Sentence]:
        for sentence in sentences:
            yield self.transform(sentence)
