from typing import Iterable, Iterator, List, Tuple
from prophetnlg import Sentence, SentenceToken, WordAnalysis
from uralicNLP import uralicApi
from uralicNLP.ud_tools import UD_node, UD_sentence
from .nltk import SentenceAnalyzer


class UralicSentenceAnalyzer(SentenceAnalyzer):
    def analyze_token(self, token: SentenceToken, **kwargs) -> SentenceToken:
        morphologies = [a[0] for a in uralicApi.analyze(token.text, self.lang)]
        return token.with_morphologies(morphologies, 'uralic')
