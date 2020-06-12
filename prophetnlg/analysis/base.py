from typing import Iterable, Iterator, List, Optional
import nltk
from prophetnlg import Sentence, SentenceToken, WordAnalysis


class TokenizerBase:
    def __init__(self, lang: str, language: str):
        self.lang = lang
        self.language = language

    def sentence_from_strings(self, strings: Iterable[str]) -> Sentence:
        return Sentence(tokens=[SentenceToken(text=s) for s in strings])


class SentenceAnalyzerBase:
    lang = ''
    language = ''
    tokenizer_class : type = TokenizerBase

    def __init__(self, **kwargs):
        self.lang = self.lang or kwargs.get('lang')
        self.language = self.language or kwargs.get('language')
        self.tokenizer = self.tokenizer_class(lang=self.lang, language=self.language)

    def analyze_token(self, token: SentenceToken, **kwargs) -> SentenceToken:
        return token

    def analyze_sentence(self, sentence: Sentence) -> Sentence:
        tokens = [self.analyze_token(token) for token in sentence.tokens]
        return sentence.replace(tokens=tokens)

    def analyze_text(self, text: str) -> Iterator[Sentence]:
        for sentence in self.tokenizer.tokenize(text):
            yield self.analyze_sentence(sentence)

    def analyze_word(self, word: str) -> Optional[SentenceToken]:
        for sentence in self.analyze_text(word):
            for token in sentence.tokens:
                return token
        return None
