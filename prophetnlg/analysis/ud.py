from typing import Iterator, List, Set
from uralicNLP import dependency, uralicApi
from uralicNLP.ud_tools import UD_node, UD_sentence
from prophetnlg import Sentence, SentenceToken, WordAnalysis
from .base import SentenceAnalyzerBase, TokenizerBase
from .udparse import ud_node_morphology


class UDTokenizer(TokenizerBase):
    def _parse_ud_node(self, node: UD_node) -> SentenceToken:
        misc = dict(f.split('=', 1) for f in node.misc.split('|') if '=' in f)
        spaces_after = ' '
        if misc.get('SpaceAfter') == 'No':
            spaces_after = ''
        if misc.get('SpacesAfter'):
            spaces_after = misc['SpacesAfter']

        spaces_after = spaces_after.replace('\\s', ' ').replace('\\n', '\n')

        token = SentenceToken(
            text=node.form,
            lang='?' if 'Foreign=Yes' in node.get_feats() else self.lang,
            spaces_after=spaces_after
        )
        m = ud_node_morphology(node)
        analysis = WordAnalysis(node.form, original=node, morphologies=[m])
        return token.with_analysis(analysis, 'ud')

    def _parse_ud_sentence(self, ud_sentence: UD_sentence) -> Sentence:
        tokens = [self._parse_ud_node(node) for node in ud_sentence]
        if tokens:
            # single whitespace after last token
            tokens[-1] = tokens[-1].replace(spaces_after=' ')
        return Sentence(tokens=tokens, formatting=True)

    def tokenize(self, text: str) -> Iterator[Sentence]:
        if not text:
            return
        for ud_sentence in dependency.parse_text(text, language=self.lang):
            yield self._parse_ud_sentence(ud_sentence)


class UDSentenceAnalyzer(SentenceAnalyzerBase):
    tokenizer_class = UDTokenizer
