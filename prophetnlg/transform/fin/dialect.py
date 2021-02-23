from collections import deque
from enum import Enum
from typing import List, Sequence
import murre
from prophetnlg import Sentence
from ..base import ConfigBase, SentenceToTextTransform


class FinDialect(str, Enum):
    POHJOISSATAKUNTA = 'Pohjois-Satakunta'
    KESKIKARJALA = 'Keski-Karjala'
    KAINUU = 'Kainuu'
    ETELAPOHJANMAA = 'Etelä-Pohjanmaa'
    ETELASATAKUNTA = 'Etelä-Satakunta'
    POHJOISSAVO = 'Pohjois-Savo'
    POHJOISKARJALA = 'Pohjois-Karjala'
    KESKIPOHJANMAA = 'Keski-Pohjanmaa'
    KAAKKOISHAME = 'Kaakkois-Häme'
    POHJOINENKESKISUOMI = 'PohjoinenKeski-Suomi'
    POHJOISPOHJANMAA = 'Pohjois-Pohjanmaa'
    POHJOINENVARSINAISSUOMI = 'PohjoinenVarsinais-Suomi'
    ETELAKARJALA = 'Etelä-Karjala'
    LANSIUUSIMAA = 'Länsi-Uusimaa'
    INKERINSUOMALAISMURTEET = 'Inkerinsuomalaismurteet'
    LANTINENKESKISUOMI = 'LäntinenKeski-Suomi'
    LANSISATAKUNTA = 'Länsi-Satakunta'
    ETELASAVO = 'Etelä-Savo'
    LANSIPOHJA = 'Länsipohja'
    POHJOISHAME = 'Pohjois-Häme'
    ETELAINENKESKISUOMI = 'EteläinenKeski-Suomi'
    ETELAHAME = 'Etelä-Häme'
    PERAPOHJOLA = 'Peräpohjola'


class FinDialectConfig(ConfigBase):
    dialect: FinDialect


class FinDialectTransform(SentenceToTextTransform):
    config_class = FinDialectConfig
    config: FinDialectConfig

    def get_text(self, sentence: Sentence) -> str:
        return murre.dialectalize_sentence(sentence.as_text(), self.config.dialect)

    def transform_sequence(self, sentences: Sequence[Sentence]) -> List[str]:
        texts = [(s.as_text(), s.passthrough) for i, s in enumerate(sentences)]
        transformed = deque(murre.dialectalize_sentences([text for text, pt in texts if not pt]))
        return [text if pt else transformed.popleft() for text, pt in texts]


class FinNormalizeDialectTransform(SentenceToTextTransform):
    def get_text(self, sentence: Sentence) -> str:
        return murre.normalize_sentence(sentence.as_text())

    def transform_sequence(self, sentences: Sequence[Sentence]) -> List[str]:
        texts = [(s.as_text(), s.passthrough) for i, s in enumerate(sentences)]
        transformed = deque(murre.normalize_sentences([text for text, pt in texts if not pt]))
        return [text if pt else transformed.popleft() for text, pt in texts]
