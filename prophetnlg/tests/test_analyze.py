from typing import Set
import unittest
from prophetnlg import SentenceToken
from prophetnlg.analysis.fin import (
    FinSentenceAnalyzer,
    FinCGSentenceAnalyzer,
    FinHeuristicSentenceAnalyzer,
    FinHeuristicUDSentenceAnalyzer,
    FinUDSentenceAnalyzer
)


class TestFinSentenceAnalyzer(unittest.TestCase):
    analysis_type = 'uralic'
    analyzer_class : type = FinSentenceAnalyzer

    def setUp(self):
        self.analyzer = self.analyzer_class()

    def test_analyze_text(self):
        sentences = list(self.analyzer.analyze_text('Kissa hyppäsi katolle.'))
        self.assertEqual(len(sentences), 1)
        self.assertEqual(len(sentences[0].tokens), 4)
        expected = [{'Kissa', 'kissa'}, {'hypätä'}, {'katto'}, {'.'}]

        # both kissa and Kissa are allowed but also only one is needed
        for token, exp in zip(sentences[0].tokens, expected):
            assert token.analyses[self.analysis_type].get_lemmas() & exp

        self.assertEqual(sentences[0].as_text(), 'Kissa hyppäsi katolle.')


class TestFinCGSentenceAnalyzer(TestFinSentenceAnalyzer):
    analysis_type = 'cg'
    analyzer_class = FinCGSentenceAnalyzer


class TestFinUDSentenceAnalyzer(TestFinSentenceAnalyzer):
    analysis_type = 'ud'
    analyzer_class = FinUDSentenceAnalyzer


class TestFinHeuristicSentenceAnalyzer(TestFinSentenceAnalyzer):
    analysis_type = 'guess'
    analyzer_class = FinHeuristicSentenceAnalyzer


class TestFinUDHeuristicAnalyzer(TestFinSentenceAnalyzer):
    analysis_type = 'guess'
    analyzer_class = FinHeuristicUDSentenceAnalyzer

    def test_analyze_unknown_nouns(self):
        text = 'Belsebyyb istui. Rokkilazerit juoksivat. Xaczhimalkin miettii.'
        sentences = list(self.analyzer.analyze_text(text))
        self.assertEqual(sentences[0].tokens[0].lemma, 'Belsebyyb')
        self.assertEqual(sentences[1].tokens[0].lemma, 'rokki#lazeri')
        self.assertEqual(sentences[2].tokens[0].lemma, 'Xaczhimal')

    def test_analyze_unknown_verb(self):
        text = 'Koira lurppasi sohvalla. Kissa huheltaisi. Pöllöt mömmertävät.'
        sentences = list(self.analyzer.analyze_text(text))
        self.assertEqual(sentences[0].tokens[1].lemma, 'lurpata')
        self.assertEqual(sentences[1].tokens[1].lemma, 'huheltaa')
        self.assertEqual(sentences[2].tokens[1].lemma, 'mömmertää')

    def test_analyze_unknown_noun_different_forms(self):
        text = '''
        Belsebyybin koira istui.
        Valitsimme Rokkiturpot eduskuntaan.
        Xaczhimalia harmitti matsin tulos.
        Mooseksena olo on raskasta.
        Kuka ryhtyisi Ismaeliksi?
        Zamuelista saatiin hassu kuningas.
        Lähdemme joukolla Mökköön.
        Belielillä on peliä.
        Sain tämän setelin Määmiltä.
        '''
        sentences = list(self.analyzer.analyze_text(text))
        self.assertEqual(sentences[0].tokens[0].lemma, 'Belsebyyb')
        self.assertEqual(sentences[1].tokens[1].lemma, 'rokki#turppo')
        self.assertEqual(sentences[2].tokens[0].lemma, 'Xaczhimal')
        self.assertEqual(sentences[3].tokens[0].lemma, 'Mooses')
        self.assertEqual(sentences[4].tokens[2].lemma, 'ismaeli')
        self.assertEqual(sentences[5].tokens[0].lemma, 'Zamuel')
        self.assertEqual(sentences[6].tokens[2].lemma, 'Mökkö')
        self.assertEqual(sentences[7].tokens[0].lemma, 'Beliel')
        self.assertEqual(sentences[8].tokens[3].lemma, 'määmä')


class TestFinAnalysisTense(unittest.TestCase):
    def setUp(self):
        self.analyzer = FinHeuristicSentenceAnalyzer()

    def test_tense_analysis(self):
        text = 'Tietäisin. Mistä tiesit? Miten olisitte voineet haluta tietää? Miten hän oli tietänyt?'
        sentences = list(self.analyzer.analyze_text(text))
        self.assertEqual(sentences[0].tokens[0].morphology, 'tietää+V+Act+Cond+Sg1')

        self.assertEqual(sentences[1].tokens[1].morphology, 'tietää+V+Act+Ind+Prt+Sg2')

        self.assertEqual(sentences[2].tokens[1].morphology, 'olla+V+Act+Cond+Pl2')
        self.assertEqual(sentences[2].tokens[2].morphology, 'voida+V+Act+PrfPrc+Pl+Nom')
        self.assertEqual(sentences[2].tokens[3].morphology, 'haluta+V+Act+InfA+Sg+Lat')
        self.assertEqual(sentences[2].tokens[4].morphology, 'tietää+V+Act+InfA+Sg+Lat')

        self.assertEqual(sentences[3].tokens[2].morphology, 'olla+V+Act+Ind+Prt+Sg3')
        self.assertEqual(sentences[3].tokens[3].morphology, 'tietää+V+Act+PrfPrc+Sg+Nom')
