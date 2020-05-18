import unittest
from prophetnlg.analysis.fin import FinSentenceAnalyzer
from prophetnlg.generator.fin import SentenceTokenGenerator


class TestFinNounGenerator(unittest.TestCase):
    def setUp(self):
        # analyzer returns lemmas only with unambiguous analyses
        self.analyzer = FinSentenceAnalyzer()
        self.generator = SentenceTokenGenerator()

    def test_generate(self):
        text = self.generator._generate('vuosi+N+Pl+Par')
        self.assertEqual(text, 'vuosia')

    def test_token_with_new_lemma_known(self):
        token = self.analyzer.analyze_word('keppien')
        new_word = self.generator.token_with_new_lemma(token, 'gouda').text
        self.assertEqual(new_word, 'goudien')

    def test_token_with_new_lemma(self):
        token = self.analyzer.analyze_word('keppien')
        new_word = self.generator.token_with_new_lemma(token, 'Truuda').text
        self.assertEqual(new_word, 'Truudien')

    def test_token_with_new_lemma_harder(self):
        token = self.analyzer.analyze_word('kasveilla')
        new_word = self.generator.token_with_new_lemma(token, 'Truuda').text
        self.assertEqual(new_word, 'Truudilla')

    def test_token_with_new_lemma_no_replacement(self):
        token = self.analyzer.analyze_word('lakkien')
        new_word = self.generator.token_with_new_lemma(token, 'Merhab').text
        self.assertEqual(new_word, 'Merhabien')

    def test_token_with_new_lemma_no_replacement_harder(self):
        token = self.analyzer.analyze_word('hipeillä')
        new_word = self.generator.token_with_new_lemma(token, 'Mozokoz').text
        self.assertEqual(new_word, 'Mozokozeilla')


class TestFinVerbGenerator(unittest.TestCase):
    def setUp(self):
        # analyzer returns lemmas only with unambiguous analyses
        self.analyzer = FinSentenceAnalyzer()
        self.generator = SentenceTokenGenerator()

    def test_generate(self):
        text = self.generator._generate('juosta+V+Act+Ind+Prt+Pl3')
        self.assertEqual(text, 'juoksivat')

    def test_token_with_new_lemma_known(self):
        token = self.analyzer.analyze_word('kepitimme')
        new_word = self.generator.token_with_new_lemma(token, 'kimmeltää').text
        self.assertEqual(new_word, 'kimmelsimme')

    def test_token_with_new_lemma(self):
        token = self.analyzer.analyze_word('kipitimme')
        new_word = self.generator.token_with_new_lemma(token, 'huheltaa').text
        self.assertEqual(new_word, 'huhelsimme')

