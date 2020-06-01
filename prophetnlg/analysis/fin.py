from typing import List, Tuple
from prophetnlg import SentenceToken, WordAnalysis
from uralicNLP.ud_tools import UD_node
from .uralic import UralicSentenceAnalyzer, UralicDepSentenceAnalyzer


class FinSentenceAnalyzer(UralicSentenceAnalyzer):
    lang = 'fin'
    language = 'finnish'


class FinDepSentenceAnalyzer(UralicDepSentenceAnalyzer):
    lang = 'fin'
    language = 'finnish'


class FinHeuristicSentenceAnalyzer(FinDepSentenceAnalyzer):
    """
    Analyze sentences by TurkuNLP, single tokens heuristically from
    TurkuNLP and uralicNLP analysis output.
    """
    POS_UD_TO_URALIC_MAPPING = {
        'C': 'CC'
    }

    def _lemma_and_prefix_from_morphology(self, morph: str) -> str:
        *prefixes, main_part = morph.split('#')
        return main_part.split('+')[0], '#'.join(prefixes)

    def _compound_from_morphology(self, morph: str) -> str:
        return '#'.join(part.split('+')[0] for part in morph.split('#'))

    def _pos_from_ud(self, ud_node: UD_node) -> str:
        if ud_node.upostag == 'VERB':
            return 'V'
        return self.POS_UD_TO_URALIC_MAPPING.get(
            ud_node.xpostag,
            ud_node.xpostag
        )

    def _morphology_from_ud(self, ud_node: UD_node) -> str:
        lemma = ud_node.lemma
        pos = self._pos_from_ud(ud_node)
        extra = ''
        if pos == 'V':
            extra = '+Act+InfA+Sg+Lat'
        elif pos != 'Adv':
            extra = '+Sg+Nom'
        return f'{lemma}+{pos}{extra}'

    def _pick_pos(self, token: SentenceToken) -> str:
        """
        1. If UD and Omorfi agree about POS, use it.
        2. Otherwise, use first Omorfi POS, if available.
        3. Otherwise, use UD POS.
        """
        morphologies = token.analyses['uralic'].analysis
        all_u_pos = [m.split('+', 2)[1] for m in morphologies]
        ud_pos = self._pos_from_ud(token.analyses['ud'].analysis)
        if ud_pos in all_u_pos:
            return ud_pos
        if len(all_u_pos):
            return all_u_pos.pop()
        return ud_pos

    def _morphology_guess_score(self, token: SentenceToken, morph: str) -> tuple:
        ud_node = token.analyses['ud'].analysis
        ud_lemma = ud_node.lemma
        lemma, pos, *_ = morph.split('+')
        word_morphologies = morph.split('#')
        compound = '#'.join(m.split('+', 1)[0] for m in word_morphologies)
        common_scored_items = (
            lemma == ud_lemma,
            compound == ud_lemma,
            lemma.lower() == ud_lemma.lower(),
            compound.lower() == ud_lemma.lower(),
            compound.lower().replace('#', '') == ud_lemma.lower().replace('#', ''),
            morph.count('Use/Hyphen') == token.text.count('-'),
        )
        return common_scored_items + (len(morph),)

    def _best_morphology(self, token: SentenceToken, morphologies: List[str]) -> str:
        return sorted(
            morphologies,
            key=lambda m: self._morphology_guess_score(token, m),
            reverse=True
        )[0]

    def _pick_morphology(self, token: SentenceToken, pos: str) -> str:
        if token.morphology:
            return token.morphology
        morphologies = token.analyses['uralic'].analysis
        morphology_parts = [m.split('#') for m in morphologies]
        morphologies_same_pos = [
            morph for morph, parts in zip(morphologies, morphology_parts)
            if len(parts[-1].split('+')) > 1 and parts[-1].split('+')[1] == pos
        ]
        if morphologies_same_pos:
            # select best one with same POS
            return self._best_morphology(token, morphologies_same_pos)
        if morphologies:
            # Omorfi and UD don't agree with POS,
            # select best one from Omorfi
            return self._best_morphology(token, morphologies)
        # completely foreign to Omorfi, just select what UD suggests
        return self._morphology_from_ud(token.analyses['ud'].analysis)

    def _pick_lemma_and_prefix(self, token: SentenceToken, morph: str) -> Tuple[str, str]:
        return self._lemma_and_prefix_from_morphology(morph)

    def _heuristic_analyze_token(self, token: SentenceToken) -> SentenceToken:
        attrs = {}
        attrs['pos'] = pos = self._pick_pos(token)
        attrs['morphology'] = morphology = self._pick_morphology(token, pos)
        attrs['lemma'], attrs['prefix_morphology'] = \
            self._pick_lemma_and_prefix(token, morphology)
        return token._replace(**attrs)

    def analyze_token(self, token: SentenceToken, *args, **kwargs) -> SentenceToken:
        analyzed_token = super().analyze_token(token, *args, **kwargs)
        return self._heuristic_analyze_token(analyzed_token)
