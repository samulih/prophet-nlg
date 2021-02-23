from enums import Enum
from typing import Dict, Mapping
from prophetnlg import ConfigBase, Sentence, SentenceToken
from prophetnlg.generator.fin import SentenceTokenGenerator
from ..base import SentenceTransformBase


class Tense(str, Enum):
    PLUPERFECT = 'PLUPERFECT'
    PERFECT = 'PERFECT'
    IMPERFECT = 'IMPERFECT'
    PRESENT = 'PRESENT'


class TenseConfig(ConfigBase):
    generator: SentenceTokenGenerator
    tense_mapping: Dict[Tense, Tense]


class TenseTransform(SentenceTransformBase):
    def can_change_verb_tense(self, token: SentenceToken) -> bool:
        if any(m in token.morphology for m in ('ConNeg', 'InfMa')):
            return False
        return True

    def replace_tense(self, sentence: Sentence, tense_mapping: Mapping[str, str]) -> Sentence:
        new_tokens = []
        for token in sentence.tokens:
            print(token)
