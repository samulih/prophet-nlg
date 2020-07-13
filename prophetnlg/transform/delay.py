import abc
from collections import deque
from typing import Any, Deque, Iterable, Optional, Union
from prophetnlg import Sentence, SentenceToken
from .base import ConfigBase, SentenceTransformBase


class SentenceDelayConfig(ConfigBase):
    buffer_size: int = 5
    buffer: Deque[Sentence] = deque()
    looping: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.buffer = deque(self.buffer, self.buffer_size)


class SentenceDelayTransform(SentenceTransformBase):
    config_class = SentenceDelayConfig
    config: SentenceDelayConfig

    def transform(self, sentence: Sentence) -> Sentence:
        if len(self.config.buffer) < self.config.buffer.maxlen:
            self.config.buffer.append(sentence)
            return Sentence()
        else:
            first = self.config.buffer[0]
            self.config.buffer.append(sentence)
            return first

    def transform_stream(self, sentences: Iterable[Sentence]) -> Iterable[Sentence]:
        yield from super().transform_stream(sentences)
        while True:
            yield from self.config.buffer
            if not self.config.looping:
                break
        self.config.buffer.clear()
