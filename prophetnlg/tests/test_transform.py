import itertools
import os
from typing import Iterator
import pytest
import unittest
from prophetnlg import SentenceToken
from prophetnlg.analysis.fin import (
    FinHeuristicSentenceAnalyzer,
    FinHeuristicUDSentenceAnalyzer
)
from prophetnlg.generator.fin import SentenceTokenGenerator
from prophetnlg.transform.replace import (
    LemmaReplaceStreamTransform,
    LemmaMapStreamTransform
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

class TestMixin:
    def setUp(self):
        self.analyzer = self.analyzer_class()
        self.generator = SentenceTokenGenerator()

    def _get_text_tokens(self, text: str) -> Iterator[SentenceToken]:
        sentences = self.analyzer.analyze_text(text)
        for sentence in sentences:
            for token in sentence.tokens:
                yield token


class TestReplaceTransform(TestMixin, unittest.TestCase):
    analyzer_class : type = FinHeuristicSentenceAnalyzer

    def test_replace_stream_transform_noun(self):
        replacements = self._get_text_tokens('Pöllöt miettivät syviä ajatuksia.')
        transform = LemmaReplaceStreamTransform(
            generator=self.generator,
            replacement_stream=replacements,
            replace_pos=['N']
        )
        source_sentence = next(self.analyzer.analyze_text('Pelle hyppäsi veteen.'))
        new_sentence = transform.transform(source_sentence)
        self.assertEqual(new_sentence.as_text(), 'Pöllö hyppäsi ajatukseen.')

    def test_replace_stream_transform_adj_noun(self):
        replacements = self._get_text_tokens('Pöllöt miettivät syviä ajatuksia.')
        transform = LemmaReplaceStreamTransform(
            generator=self.generator,
            replacement_stream=replacements,
            replace_pos=['N', 'A']
        )
        source_sentence = next(self.analyzer.analyze_text('Pelle hyppäsi hassun aamun sarastaessa.'))
        new_sentence = transform.transform(source_sentence)
        self.assertEqual(new_sentence.as_text(), 'Pöllö hyppäsi syvän ajatuksen sarastaessa.')

    def test_transform_bible_to_jokes(self):
        bible = os.path.join(DATA_DIR, 'vt_1moos_1_2.txt')
        jokes = os.path.join(DATA_DIR, 'vitsit.txt')
        with open(jokes) as j, open(bible) as b:
            text = b.read()
            # TurkuNLP parses ________ as a noun
            joke_text = j.read().replace('_', '')
            replacements = self._get_text_tokens(joke_text)

        transform = LemmaReplaceStreamTransform(
            generator=self.generator,
            replacement_stream=replacements,
            replace_pos=['N', 'A']
        )
        sentences = self.analyzer.analyze_text(text)
        result = [transform.transform(s).as_text() for s in itertools.islice(sentences, 5)]
        self.assertEqual(result,
            [
                'Erämaassa loi kahdeksikko kuumuuden ja jutun.',
                'Ja sveitsinjuusto oli kumma ja vanha, ja reikä oli juuston päällä, ja tuuletuksen kerrostalo liikkui seinien päällä.',
                'Ja ilmoitus sanoi: "tulkoon hissi".',
                'Ja naapuritalo tuli.',
                'Ja hissi näki, että hissi oli lähin; ja ovi erotti hissin puolesta.'
            ]
        )


class TestMapTransform(TestMixin, unittest.TestCase):
    analyzer_class : type = FinHeuristicSentenceAnalyzer

    def test_map_stream_transform(self):
        replacements = self._get_text_tokens('Pöllöt miettivät syviä ajatuksia aamulla.')
        transform = LemmaMapStreamTransform(
            generator=self.generator,
            replacement_stream=replacements,
            replace_pos=['N', 'A']
        )
        source_sentence = next(self.analyzer.analyze_text('Jarno hyppäsi veteen - veden pintaa ui hassu Jarno.'))
        new_sentence = transform.transform(source_sentence)
        self.assertEqual(new_sentence.as_text(), 'Pöllö hyppäsi ajatukseen - ajatuksen aamua ui syvä pöllö.')

    def test_transform_bible_to_jokes(self):
        bible = os.path.join(DATA_DIR, 'vt_1moos_1_2.txt')
        jokes = os.path.join(DATA_DIR, 'vitsit.txt')
        with open(jokes) as j, open(bible) as b:
            text = b.read()
            # TurkuNLP parses ____ as a noun
            joke_text = j.read().replace('_', '')
            replacements = self._get_text_tokens(joke_text)

        transform = LemmaMapStreamTransform(
            generator=self.generator,
            replacement_stream=replacements,
            replace_pos=['N', 'A']
        )
        sentences = self.analyzer.analyze_text(text)
        result = [transform.transform(s).as_text() for s in itertools.islice(sentences, 5)]
        self.assertEqual(result,
            [
                'Erämaassa loi kahdeksikko kuumuuden ja jutun.',
                'Ja juttu oli kumma ja vanha, ja sveitsinjuusto oli reiän päällä, ja kahdeksikon juusto liikkui tuuletusten päällä.',
                'Ja kahdeksikko sanoi: "tulkoon kerrostalo".',
                'Ja kerrostalo tuli.',
                'Ja kahdeksikko näki, että kerrostalo oli lähin; ja kahdeksikko erotti kerrostalon sveitsinjuustosta.'
            ]
        )

@pytest.mark.slow
class TestReplaceTransformUD(TestReplaceTransform):
    analyzer_class = FinHeuristicUDSentenceAnalyzer


@pytest.mark.slow
class TestMapTransformUD(TestMapTransform):
    analyzer_class = FinHeuristicUDSentenceAnalyzer
