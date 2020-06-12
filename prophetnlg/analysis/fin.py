from .cg import CGSentenceAnalyzer
from .heuristic import HeuristicSentenceAnalyzer, HeuristicUDSentenceAnalyzer
from .uralic import UralicSentenceAnalyzer
from .ud import UDSentenceAnalyzer


class FinMixin:
    lang = 'fin'
    language = 'finnish'


class FinSentenceAnalyzer(FinMixin, UralicSentenceAnalyzer):
    pass


class FinCGSentenceAnalyzer(FinMixin, CGSentenceAnalyzer):
    pass


class FinUDSentenceAnalyzer(FinMixin, UDSentenceAnalyzer):
    pass


class FinHeuristicSentenceAnalyzer(FinMixin, HeuristicSentenceAnalyzer):
    pass


class FinHeuristicUDSentenceAnalyzer(FinMixin, HeuristicUDSentenceAnalyzer):
    pass
