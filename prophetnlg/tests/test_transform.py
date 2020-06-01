from typing import Iterator
import unittest
from prophetnlg import SentenceToken
from prophetnlg.analysis.fin import (
    FinSentenceAnalyzer,
    FinHeuristicSentenceAnalyzer
)
from prophetnlg.generator.fin import SentenceTokenGenerator
from prophetnlg.transform.replace import LemmaReplaceStreamTransform


class TestReplaceTransform(unittest.TestCase):
    def setUp(self):
        self.analyzer = FinHeuristicSentenceAnalyzer()
        self.generator = SentenceTokenGenerator()

    def _get_text_tokens(self, text: str) -> Iterator[SentenceToken]:
        sentences = self.analyzer.analyze_text(text)
        for sentence in sentences:
            for token in sentence.tokens:
                yield token

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
