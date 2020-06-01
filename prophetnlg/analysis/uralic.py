from typing import Iterable, Iterator, List, Tuple
from prophetnlg import Sentence, SentenceToken, WordAnalysis
from uralicNLP import cg3, dependency, uralicApi
from uralicNLP.ud_tools import UD_node, UD_sentence
from .base import SentenceAnalyzer

Cg3Token = Tuple[str, List[cg3.Cg3Word]]


class UralicSentenceAnalyzer(SentenceAnalyzer):
    def __init__(self):
        self.cg = cg3.Cg3(self.lang)

    def disambiguate(self, words: Iterable[str]) -> List[Cg3Token]:
        words = list(words)
        cg_tokens = self.cg.disambiguate(words)
        assert len(words) == len(cg_tokens)
        return cg_tokens

    def analyze_sentence(self, sentence: Sentence) -> Sentence:
        tokens = list(sentence.tokens)
        disambiguated = self.disambiguate([t.text for t in tokens])
        tokens = [self.analyze_token(t, d) for t, d in zip(tokens, disambiguated)]
        return sentence._replace(tokens=tokens)

    def _cg_token_lemma_morphology_pos(self, cg_token: Cg3Token) -> Tuple[str, str, str]:
        lemma, cg_words = cg_token
        # TODO: use cg only in case of single match and resolve later?
        #if len(cg_words) > 1:
        #    return '', '', ''
        lemma, morphology = cg_words[0].lemma, cg_words[0].morphology
        return lemma, '+'.join(m for m in [lemma] + morphology if '<' not in m), morphology[0]

    def analyze_token(self, token: SentenceToken, cg_token: Cg3Token = None) -> SentenceToken:
        # analyze via uralic analyzer & disambiguate with CG

        morphologies = [a[0] for a in uralicApi.analyze(token.text, language=self.lang)]
        token = token.with_analysis(morphologies, 'uralic').with_analysis(cg_token, 'cg')

        cg_lemma, cg_morphology, cg_pos = self._cg_token_lemma_morphology_pos(cg_token)
        if cg_morphology in morphologies:
            # disambiguation returned a viable alternative
            # use CG lemma & morphology
            return token._replace(
                morphology=cg_morphology,
                lemma=cg_lemma,
                pos=cg_pos
            )

        # use uralic lemma & morphology if non-ambiguous

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
