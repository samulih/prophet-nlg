import unittest
from prophetnlg import Sentence, SentenceToken
from prophetnlg.analysis.fin import FinHeuristicSentenceAnalyzer
from prophetnlg.transform.filter import (
    SequentialTokenFilterTransform,
    SequentialTokenFilterByPosTransform,
    StochasticTokenFilterTransform,
    StochasticTokenFilterByPosTransform
)


def get_sentence(text: str) -> Sentence:
    analyzer = FinHeuristicSentenceAnalyzer()
    return list(analyzer.analyze_text(text))[0]


class TestSequentialTokenFilterTransform(unittest.TestCase):
    def test_sequential_filter(self):
        sentence = get_sentence('Juokse sinä humma kun tuo taivas on niin tumma!')
        f1 = SequentialTokenFilterTransform(effect=0.333)
        f2 = SequentialTokenFilterTransform(effect=0.3334)
        sentence_f1 = f1.transform(sentence)
        sentence_f2 = f2.transform(sentence)
        self.assertEqual(
            [t.passthrough for t in sentence_f1.tokens],
            [1, 1, 0, 1, 1, 0, 1, 1, 0, 1]
        )
        self.assertEqual(
            [t.passthrough for t in sentence_f2.tokens],
            [1, 0, 1, 1, 0, 1, 1, 0, 1, 1]
        )

    def test_sequential_filter_nested(self):
        sentence = get_sentence('Juokse sinä humma kun tuo taivas on niin tumma!')
        f1 = SequentialTokenFilterTransform(effect=0.50001)
        f2 = SequentialTokenFilterTransform(effect=0.50001)
        sentence_f1 = f1.transform(sentence)
        sentence_f2 = f2.transform(sentence_f1)
        self.assertEqual(
            [t.passthrough for t in sentence_f1.tokens],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        )
        self.assertEqual(
            [t.passthrough for t in sentence_f2.tokens],
            [0, 2, 1, 2, 0, 2, 1, 2, 0, 2]
        )


class TestStochasticTokenFilterTransform(unittest.TestCase):
    def test_stochastic_filter(self):
        sentence = get_sentence('juokse sinä juu joo ' * 25)
        f1 = StochasticTokenFilterTransform(effect=0.5)
        f2 = StochasticTokenFilterTransform(effect=0.5)
        sentence_f1 = f1.transform(sentence)
        sentence_f2 = f2.transform(sentence_f1)
        passthrough_f1 = [t.passthrough == 1 for t in sentence_f1.tokens]
        passthrough_f2 = [t.passthrough == 2 for t in sentence_f2.tokens]
        assert 40 < sum([t.passthrough == 1 for t in sentence_f1.tokens]) < 60
        assert 15 < sum([t.passthrough == 1 for t in sentence_f2.tokens]) < 35
        assert 40 < sum([t.passthrough == 2 for t in sentence_f2.tokens]) < 60
        assert all(t.passthrough in (0, 1) for t in sentence_f1.tokens)
        assert all(t.passthrough in (0, 1, 2) for t in sentence_f2.tokens)
