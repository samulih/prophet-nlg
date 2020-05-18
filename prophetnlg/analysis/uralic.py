from typing import Iterator
from prophetnlg import Sentence, SentenceToken, WordAnalysis
from uralicNLP import uralicApi, dependency
from uralicNLP.ud_tools import UD_node, UD_sentence
from .base import SentenceAnalyzer


class UralicSentenceAnalyzer(SentenceAnalyzer):
    def analyze_token(self, token: SentenceToken) -> SentenceToken:
        morphologies = [
            a[0] for a in uralicApi.analyze(token.text, language=self.lang)
        ]
        token = token.with_analysis(morphologies, 'uralic')
        morphologies = set(morphologies)
        lemmas = {m.split('+', 1)[0] for m in morphologies}
        lemmas_cap = {l.capitalize() for l in lemmas}
        pos = {m.split('+', 2)[1] for m in morphologies}
        lemma = lemmas.pop() if len(lemmas) == 1 else ''
        if not lemma:
            lemma = lemmas_cap.pop() if len(lemmas_cap) == 1 else ''
        return token._replace(
            morphology=morphologies.pop() if len(morphologies) == 1 else '',
            lemma=lemma,
            pos=pos.pop() if len(pos) == 1 else ''
        )
        return token


class UralicDepSentenceAnalyzer(UralicSentenceAnalyzer):
    def _parse_ud_node(self, node: UD_node) -> SentenceToken:
        spaces_after = ' '
        if 'SpaceAfter=No' in node.misc:
            spaces_after = ''
        if 'SpacesAfter=' in node.misc:
            spaces_after = node.misc.split('=')[1]

        spaces_after = spaces_after.replace('\\s', ' ').replace('\\n', '\n')

        token = SentenceToken(
            text=node.form,
            lang='?' if 'Foreign=Yes' in node.get_feats() else self.lang,
            spaces_after=spaces_after
        )
        return token.with_analysis(node, 'ud')

    def _parse_ud_sentence(self, ud_sentence: UD_sentence) -> Sentence:
        tokens = [self._parse_ud_node(node) for node in ud_sentence]
        return Sentence(tokens=tokens)

    def tokenize(self, text: str) -> Iterator[Sentence]:
        if not text:
            return
        for ud_sentence in dependency.parse_text(text, language=self.lang):
            yield self._parse_ud_sentence(ud_sentence)
