from typing import Iterator, List
import nltk
from prophetnlg import Sentence, SentenceToken, WordAnalysis


class SentenceAnalyzer:
    def _parse_string_tokens(self, strings: List[str]) -> Sentence:
        return Sentence(tokens=[SentenceToken(text=s) for s in strings])

    def tokenize(self, text: str) -> Iterator[Sentence]:
        sentence_texts = nltk.sent_tokenize(text, language=self.language)
        for sentence_text in sentence_texts:
            tokens = nltk.word_tokenize(sentence_text, language=self.language)
            yield self._parse_string_tokens(tokens)

    def analyze_token(self, token: SentenceToken) -> SentenceToken:
        return token

    def analyze_sentence(self, sentence: Sentence) -> Sentence:
        tokens = [self.analyze_token(token) for token in sentence.tokens]
        return sentence._replace(tokens=tokens)

    def analyze_text(self, text: str) -> Iterator[Sentence]:
        for sentence in self.tokenize(text):
            yield self.analyze_sentence(sentence)

    def analyze_word(self, word: str) -> SentenceToken:
        for sentence in self.analyze_text(word):
            for token in sentence.tokens:
                return token
