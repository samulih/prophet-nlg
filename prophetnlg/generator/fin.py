import os
from pathlib import Path
import sqlite3
from typing import Tuple, Union
from uralicNLP import uralicApi, semfi
from prophetnlg import SentenceToken

cache_path = Path(uralicApi.__where_models('fin')) / 'cache.db'
semfi_db = semfi.__get_connection('fin')
cache_db = sqlite3.connect(cache_path.as_posix())


def get_lemma_text(lemma_or_token: Union[str, SentenceToken]) -> str:
    return lemma_or_token if isinstance(lemma_or_token, str) else lemma_or_token.lemma


def get_last_part_morphology(token: SentenceToken) -> str:
    if '#' in token.morphology:
        return token.morphology[token.morphology.rindex('#'):]
    else:
        return token.morphology


def common_suffix(*words: str) -> str:
    return os.path.commonprefix([w[::-1] for w in words])[::-1]


def find_similar_and_suffix(word: str, pos: str) -> Tuple[str, str]:
    c = cache_db.cursor()
    words = [word[idx:] for idx in range(len(word))]
    placeholders = ','.join('?' for char in word)
    sql = f'SELECT word, suffix FROM suffixes WHERE pos = ? AND suffix in ({placeholders}) ORDER BY length(suffix) DESC LIMIT 1'
    c.execute(sql, [pos] + words)
    result = c.fetchone()
    if result:
        return result
    else:
        # no match for suffix, so just return a random
        # known foreign word with no stem changes
        return ('Mikael', '')


class SentenceTokenGenerator:
    language = 'finnish'
    lang = 'fin'

    def __init__(self):
        self.similar_cache = {}

    def _generate(self, analysis: str, similar_token: SentenceToken = None) -> str:
        results = uralicApi.generate(analysis, language=self.lang)
        if results:
            words = [r[0] for r in results]
            if similar_token:
                # return the word that ends most similarly to reference token
                return sorted(
                    words,
                    key=lambda w: len(common_suffix(w, similar_token.text)),
                    reverse=True
                )[0]
            else:
                # for example, Jari+N+Prop+Sg+Nom returns [Jari, Jarin]
                return words[-1]

    def _replace_token_lemma(
        self,
        token: SentenceToken,
        new_lemma: str
    ) -> SentenceToken:
        morphology = get_last_part_morphology(token)
        morphology_parts = morphology.split('+')
        morphology_parts[0] = new_lemma
        if new_lemma == 'laki':
            morphology_parts.insert(1, 'Hom2')
        # filter out semantic annotations for morphological generation
        morphology_parts = [p for p in morphology_parts if not p.startswith('Sem/')]
        new_morphology = '+'.join(morphology_parts)
        if new_lemma[0].isupper():
            # starts with upper case, treat as a proper noun
            if '+N+Prop+' not in new_morphology:
                new_morphology = new_morphology.replace('+N+', '+N+Prop+')
        else:
            # starts with lower case, treat as a common noun
            if '+N+Prop+' in new_morphology:
                new_morphology = new_morphology.replace('+N+Prop+', '+N+')
        new_word = self._generate(new_morphology, similar_token=token)
        return token._replace(
            text=new_word,
            lemma=new_lemma,
            morphology=new_morphology,
            analyses=[]
        )

    def _inflect_lemma_like(
        self,
        lemma: str,
        inflect_token: SentenceToken
    ) -> str:
        inflected = inflect_token.text
        common_part = os.path.commonprefix([inflect_token.lemma, inflected])
        difference_length = len(inflect_token.lemma) - len(common_part)
        inflection_length = len(inflected) - len(common_part)
        return lemma[:len(lemma) - difference_length] + \
            inflected[len(inflected) - inflection_length:]

    def token_with_new_lemma(
        self,
        token: SentenceToken,
        new_lemma: Union[str, SentenceToken]
    ) -> SentenceToken:
        new_lemma = get_lemma_text(new_lemma)
        new_token = self._replace_token_lemma(token, new_lemma)
        if new_token.text:
            # successful morphological generation, return result
            return new_token
        # generation failed, so find a similar word, inflect it instead
        # and replace the stem
        similar_lemma, suffix = find_similar_and_suffix(new_lemma, token.pos)
        similar_token = self._replace_token_lemma(token, similar_lemma)
        new_text = self._inflect_lemma_like(new_lemma, similar_token)
        return new_token._replace(text=new_text)
