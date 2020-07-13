from __future__ import annotations
from typing import Any, Generic, Dict, List, Optional
import nltk
from pydantic import BaseModel


class DataClassMixin:
    def replace(self, **attrs):
        data = dict(self.dict(), **attrs)
        return type(self).construct(self.__fields_set__, **data)


class WordAnalysis(BaseModel, DataClassMixin):
    text: str = ''
    original: Optional[Any] = None
    morphologies: Optional[List[str]] = None

    def get_morphologies(self):
        return set(m for m in self.morphologies if '+' in m)

    def get_lemmas(self):
        return {m.split('+', 1)[0] for m in self.morphologies if '+' in m}

    def get_pos(self):
        return {m.split('+', 2)[1] for m in self.morphologies if '+' in m}

    @property
    def morphology(self):
        morphologies = self.get_morphologies()
        return next(iter(morphologies)) if len(morphologies) == 1 else ''

    @property
    def lemma(self):
        lemmas = self.get_lemmas()
        return next(iter(lemmas)) if len(lemmas) == 1 else ''

    @property
    def pos(self):
        pos = self.get_pos()
        return next(iter(pos)) if len(pos) == 1 else ''


class SentenceToken(BaseModel, DataClassMixin):
    text: str = ''
    lang: str = ''
    analyses: Dict[str, WordAnalysis] = {}
    spaces_after: str = ' '
    cap: bool = False
    passthrough: bool = False

    def with_analysis(self, analysis: WordAnalysis, analysis_type: str) -> SentenceToken:
        analyses = dict(self.analyses, **{analysis_type: analysis})
        return self.replace(analyses=analyses)

    def with_morphologies(self, morphologies: List[str], analysis_type: str) -> SentenceToken:
        analysis = WordAnalysis(text=self.text, original=None, morphologies=morphologies)
        return self.with_analysis(analysis, analysis_type)

    @property
    def morphology(self):
        a = self.analyses.get('guess')
        return a.morphology if a else ''

    @property
    def lemma(self):
        a = self.analyses.get('guess')
        return a.lemma if a else ''

    @property
    def pos(self):
        a = self.analyses.get('guess')
        return a.pos if a else ''



detokenizer = nltk.tokenize.treebank.TreebankWordDetokenizer()


class Sentence(BaseModel, DataClassMixin):
    tokens: List[SentenceToken] = []
    formatting: bool = False
    passthrough: int = 0

    def as_text(self):
        if self.formatting:
            s = ''.join(
                f'{t.text.capitalize() if t.cap else t.text}{t.spaces_after}'
                for t in self.tokens
            )
        else:
            s = detokenizer.detokenize([t.text for t in self.tokens]).replace('``', ' "')
        return s.capitalize().strip()
