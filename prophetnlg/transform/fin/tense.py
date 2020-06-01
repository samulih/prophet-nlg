from typing import Mapping
from prophetnlg import Sentence, SentenceToken
from prophetnlg.generator.fin import SentenceTokenGenerator


class TenseTransform:
    def __init__(self, generator: SentenceTokenGenerator):
        self.generator = generator

    def can_change_verb_tense(self, token: SentenceToken):
        if any(m in token.morphology for m in ('ConNeg', 'InfMa')):
            return False
        return True

    def replace_tense(self, sentence: Sentence, tense_mapping: Mapping[str, str]) -> Sentence:
        new_tokens = []
        for token in sentence.tokens:
            print(token)
