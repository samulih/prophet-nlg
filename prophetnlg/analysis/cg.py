import difflib
from typing import Iterable, Iterator, List, Optional, Tuple
from prophetnlg import Sentence, SentenceToken, WordAnalysis
from uralicNLP import cg3, dependency, uralicApi
from uralicNLP.ud_tools import UD_node, UD_sentence
from .nltk import SentenceAnalyzer

Cg3Token = Tuple[str, List[cg3.Cg3Word]]


def get_matched_indexes(a: List[str], b: List[str]) -> Iterator[Tuple[int, int]]:
    s = difflib.SequenceMatcher(None, a, b)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'equal':
            yield i1, i2


class CGSentenceAnalyzer(SentenceAnalyzer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cg = cg3.Cg3(self.lang)

    def disambiguate(self, words: Iterable[str]) -> List[Cg3Token]:
        words = list(words)
        cg_tokens = self.cg.disambiguate(words)

        # CG disambiguation may return tokens that are not part of original,
        # so let's remove them!
        if len(cg_tokens) > len(words):
            new_tokens = []
            for i1, i2 in get_matched_indexes([t[0] for t in cg_tokens], words):
                new_tokens.extend(cg_tokens[i1:i2])
            cg_tokens = new_tokens

        assert len(cg_tokens) == len(words)
        return cg_tokens

    def analyze_sentence(self, sentence: Sentence) -> Sentence:
        disambiguated = self.disambiguate(t.text for t in sentence.tokens)
        tokens = [self.analyze_token(t, d) for t, d in zip(sentence.tokens, disambiguated)]
        return sentence.replace(tokens=tokens)

    def _cg_token_morphologies(self, cg_token: Cg3Token = None) -> Iterator[str]:
        if not cg_token:
            return
        text, cg_words = cg_token
        for cg_word in cg_words:
            lemma, morphology_tags = cg_word.lemma, cg_word.morphology
            if morphology_tags[0] == '?':
                continue
            yield '+'.join(m for m in [lemma] + morphology_tags if '<' not in m)

    def analyze_token(self, token: SentenceToken, cg_token : Cg3Token = None, **kwargs) -> SentenceToken:
        morphologies = list(self._cg_token_morphologies(cg_token))
        return token.with_morphologies(morphologies, 'cg')
