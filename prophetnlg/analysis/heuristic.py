from typing import List, Set, Tuple
from prophetnlg import Sentence, SentenceToken, WordAnalysis
from .cg import CGSentenceAnalyzer
from .nltk import SentenceAnalyzer
from .ud import UDSentenceAnalyzer
from .uralic import UralicSentenceAnalyzer


def pick_morphology(text, *sources : List[str]) -> str:
    all_morphologies = list(set().union(*sources))
    parts = [m.split() for m in all_morphologies]
    in_sources = [[(m in s) for s in sources] for m in all_morphologies]
    morphologies_in_sources = zip(all_morphologies, in_sources, parts)

    sorted_morphologies = sorted(
        morphologies_in_sources,
        key=lambda m_i_p: (
            m_i_p[1],                              # present in many sources
            m_i_p[2][0] == m_i_p[2][0].lower(),   # lemma is lowercase
            m_i_p[0].count('Use/Hyphen') == text.count('-'), # annotated dash counts
        ),
        reverse=True
    )
    return sorted_morphologies[0][0]


class HeuristicMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cg_analyzer = CGSentenceAnalyzer(lang=self.lang, language=self.language)
        self.uralic_analyzer = UralicSentenceAnalyzer(lang=self.lang, language=self.language)

    def get_morphology(self, token: SentenceToken) -> str:
        morphologies = self.get_morphologies_by_sources(token)
        common = set().intersection(*morphologies)
        if len(common) == 1:
            return common.pop()
        return pick_morphology(token.text, common, *morphologies)

    def analyze_token(self, token: SentenceToken, **kwargs) -> SentenceToken:
        morphology = self.get_morphology(token)
        lemma, pos, *_ = morphology.split('+')
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
