from typing import Iterator, List
import nltk
from prophetnlg import Sentence, SentenceToken, WordAnalysis
from .base import TokenizerBase, SentenceAnalyzerBase


class TextTokenizer(TokenizerBase):
    def tokenize(self, text: str) -> Iterator[Sentence]:
        sentence_texts = nltk.sent_tokenize(text, language=self.language)
        for sentence_text in sentence_texts:
            strings = nltk.word_tokenize(sentence_text, language=self.language)
            yield self.sentence_from_strings(strings)


class SentenceAnalyzer(SentenceAnalyzerBase):
    tokenizer_class = TextTokenizer
