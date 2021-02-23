from itertools import chain
from typing import Dict, List, Set
from prophetnlg import Sentence, SentenceToken
from prophetnlg.datasets.semfi import SemFi
from .cg import CGSentenceAnalyzer
from .frequency import FrequencySentenceAnalyzerBase
from .heuristic import HeuristicSentenceAnalyzer, HeuristicUDSentenceAnalyzer
from .nltk import SentenceAnalyzer
from .uralic import UralicSentenceAnalyzer
from .ud import UDSentenceAnalyzer


class FinMixin:
    lang = 'fin'
    language = 'finnish'


class FinHeuristicMixin(FinMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frequency_analyzer = FinFrequencyAnalyzer()

    def get_weights(self, token: SentenceToken) -> Dict[str, float]:
        if 'semfi' in token.analyses:
            return token.analyses['semfi'].get_morphologies()
        return {}

    def analyze_sentence(self, sentence: Sentence) -> Sentence:
        sentence = super().analyze_sentence(sentence)
        return self.frequency_analyzer.analyze_sentence(sentence)

    def get_morphologies_by_sources(self, token: SentenceToken) -> List[Set[str]]:
        morphologies = super().get_morphologies_by_sources(token)
        if 'semfi' in token.analyses:
            return morphologies + token.analyses['semfi'].get_morphologies()
        return morphologies


class FinNLTKSentenceAnalyzer(FinMixin, SentenceAnalyzer):
    pass


class FinSentenceAnalyzer(FinMixin, UralicSentenceAnalyzer):
    pass


class FinCGSentenceAnalyzer(FinMixin, CGSentenceAnalyzer):
    pass


class FinUDSentenceAnalyzer(FinMixin, UDSentenceAnalyzer):
    pass


class FinHeuristicSentenceAnalyzer(FinHeuristicMixin, HeuristicSentenceAnalyzer):
    pass


class FinHeuristicUDSentenceAnalyzer(FinHeuristicMixin, HeuristicUDSentenceAnalyzer):
    pass


class FinFrequencyAnalyzer(FinMixin, FrequencySentenceAnalyzerBase):
    analysis_type = 'semfi'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.semfi = SemFi()

    def get_weights(self, token: SentenceToken) -> Dict[str, float]:
        morphologies = chain(*(t.get_morphologies() for t in token.analyses.values()))
        lemmas_pos = {tuple(m.split('+', 2)[:2]) for m in morphologies}
        if not len(lemmas_pos):
            return {}
        return self.semfi.get_frequencies(lemmas_pos)
