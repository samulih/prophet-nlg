from enum import Enum
from typing import Dict, Optional
import numpy as np
from pydantic import BaseModel, Extra
from prophetnlg import Sentence, SentenceToken
from .annotate import IncTokenPassThroughTransform
from .base import ConfigBase


class RandomState(BaseModel):
    seed: Optional[int] = None
    state: tuple = ()


class EffectConfig(ConfigBase):
    effect: float = 0.0

    def get_effect(self, token: SentenceToken) -> float:
        return self.effect


class EffectMapConfig(ConfigBase):
    effect_map: Dict[str, float] = {}
    category_attr: str = 'pos'

    def get_token_category(self, token: SentenceToken) -> str:
        return getattr(token, self.category_attr)

    def get_effect(self, token: SentenceToken) -> float:
        category = self.get_token_category(token)
        return self.effect_map.get(category, 0.0)


class SequentialConfig(ConfigBase):
    counter: int = 1

    def reset(self):
        self.counter = 1


class StochasticConfig(SequentialConfig):
    counter: int = 1
    repeatable: bool = False
    random: RandomState = RandomState()

    def reset(self):
        self.counter = 1
        self.random = RandomState(self.random_state.seed)


class SequentialTokenFilterConfig(SequentialConfig, EffectConfig):
    pass


class SequentialTokenPosFilterConfig(SequentialConfig, EffectMapConfig):
    pass


class StochasticTokenFilterConfig(StochasticConfig, EffectConfig):
    pass


class StochasticTokenPosFilterConfig(StochasticConfig, EffectMapConfig):
    pass


class SequentialTokenFilterTransformBase(IncTokenPassThroughTransform):
    config_class = SequentialTokenFilterConfig
    config: SequentialTokenFilterConfig

    def passthrough_token(self, token: SentenceToken) -> int:
        if token.passthrough:
            return 1
        counter = self.config.counter
        effect = self.config.get_effect(token)
        prev_iter = counter * effect
        this_iter = (counter + 1) * effect
        self.config.counter += 1
        return int(int(prev_iter) == int(this_iter))


class StochasticTokenFilterTransformBase(IncTokenPassThroughTransform):
    config_class = StochasticTokenFilterConfig
    config: StochasticTokenFilterConfig

    def passthrough_token(self, token: SentenceToken) -> int:
        if token.passthrough:
            return 1
        self.config.counter += 1
        random = self.config.random if self.config.repeatable else np.random
        return int(self.config.get_effect(token) < random.rand())


class SequentialTokenFilterTransform(SequentialTokenFilterTransformBase):
    pass


class StochasticTokenFilterTransform(StochasticTokenFilterTransformBase):
    pass


class SequentialTokenFilterByPosTransform(SequentialTokenFilterTransformBase):
    pass


class StochasticTokenFilterByPosTransform(StochasticTokenFilterTransformBase):
    pass
