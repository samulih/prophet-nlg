import unittest
from prophetnlg import Sentence, SentenceToken
from prophetnlg.analysis.fin import FinHeuristicSentenceAnalyzer
from prophetnlg.transform.annotate import (
    IncTokenPassThroughTransform,
    DecTokenPassThroughTransform
)


class TestTokenPassthroughTransform(unittest.TestCase):
    def get_example_sentence(self) -> Sentence:
        analyzer = FinHeuristicSentenceAnalyzer()
        return list(analyzer.analyze_text('On ilo testata.'))[0]

    def test_inc_token_passthrough(self):
        sentence = self.get_example_sentence()
        inc = IncTokenPassThroughTransform()
        # not passthrough by default
        assert all(not t.passthrough for t in sentence.tokens)
        # passthrough when transformed
        assert all(t.passthrough for t in inc.transform(sentence).tokens)
        # no side-effects on previous transform
        assert all(not t.passthrough for t in sentence.tokens)
        # nested passthrough when transformed twice
        sentence2 = inc.transform(inc.transform(sentence))
        assert all(t.passthrough == 2 for t in sentence2.tokens)

    def test_inc_dec_token_passthrough(self):
        sentence = self.get_example_sentence()
        inc = IncTokenPassThroughTransform()
        dec = DecTokenPassThroughTransform()
        # no passthrough when transformed
        assert all(not t.passthrough for t in dec.transform(sentence).tokens)
        # no side-effects on previous transform
        assert all(not t.passthrough for t in sentence.tokens)
        # nested passthrough when transformed twice
        sentence1 = inc.transform(sentence)
        sentence2 = inc.transform(sentence1)
        assert all(t.passthrough == 1 for t in sentence1.tokens)
        assert all(t.passthrough == 2 for t in sentence2.tokens)
        # properly decremented nested passthrough
        sentence_dec1 = dec.transform(sentence2)
        sentence_dec2 = dec.transform(sentence_dec1)
        assert all(t.passthrough == 1 for t in sentence_dec1.tokens)
        assert all(t.passthrough == 0 for t in sentence_dec2.tokens)
