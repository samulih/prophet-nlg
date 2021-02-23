from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple, Union
from prophetnlg import Sentence, SentenceToken, WordAnalysis
from .cg import CGSentenceAnalyzer
from .nltk import SentenceAnalyzer
from .ud import UDSentenceAnalyzer
from .uralic import UralicSentenceAnalyzer


WordFitFunction = Callable[
    [str, Tuple[str, Sequence[bool], Sequence[str]]],
    #text  morphologies   in_sources  parts_of_lemma
    Sequence[Union[bool, int, float]]
    # these tuples are ultimately sorted and largest of them will be the "guess"
]

class HeuristicMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cg_analyzer = CGSentenceAnalyzer(lang=self.lang, language=self.language)
        self.uralic_analyzer = UralicSentenceAnalyzer(lang=self.lang, language=self.language)

    def pick_morphology(self, token: SentenceToken, *sources: Iterable[str]) -> str:
        all_morphologies : List[str] = list(set().union(*sources))
        text = token.text
        if not all_morphologies:
            return f'{text}+Unkwn'
        parts = [m.split() for m in all_morphologies]
        in_sources = [[(m in s) for s in sources] for m in all_morphologies]
        morphologies_in_sources = zip(all_morphologies, in_sources, parts)

        weights = self.get_weights(token)
        sorted_morphologies = sorted(
            morphologies_in_sources,
            key=self.heuristic_fit_function(text, weights),
            reverse=True
        )
        return sorted_morphologies[0][0]

    def heuristic_fit_function(self, text, weights: Dict[str, float]) -> WordFitFunction:
        hyphen_count = text.count('-')
        return lambda m_i_p: (
            # present in many sources
            m_i_p[1],
            # lemma is lowercase
            m_i_p[2][0] == m_i_p[2][0].lower(),
            # dash annotations match the lemma
            m_i_p[0].count('Use/Hyphen') == hyphen_count,
            # greatest weight
            weights.get(m_i_p[2][0], 0.0),
        )

    def get_morphology(self, token: SentenceToken) -> str:
        morphologies = self.get_morphologies_by_sources(token)
        common = set.intersection(*morphologies)
        if len(common) == 1:
            return common.pop()
        return self.pick_morphology(token, common, *morphologies)

    def analyze_token(self, token: SentenceToken, **kwargs) -> SentenceToken:
        morphology = self.get_morphology(token)
        return token.with_morphologies([morphology], 'guess')

    def analyze_sentence(self, sentence: Sentence) -> Sentence:
        sentence = self.cg_analyzer.analyze_sentence(sentence)
        sentence = self.uralic_analyzer.analyze_sentence(sentence)
        return super().analyze_sentence(sentence)


class HeuristicSentenceAnalyzer(HeuristicMixin, SentenceAnalyzer):
    def get_morphologies_by_sources(self, token: SentenceToken) -> List[Set[str]]:
        cg_morphologies = token.analyses['cg'].get_morphologies()
        u_morphologies = token.analyses['uralic'].get_morphologies()
        return [cg_morphologies, u_morphologies]


class HeuristicUDSentenceAnalyzer(HeuristicMixin, UDSentenceAnalyzer):
    def get_morphologies_by_sources(self, token: SentenceToken) -> List[Set[str]]:
        cg_morphologies = token.analyses['cg'].get_morphologies()
        u_morphologies = token.analyses['uralic'].get_morphologies()
        ud_morphologies = token.analyses['ud'].get_morphologies()
        return [cg_morphologies, u_morphologies, ud_morphologies]
