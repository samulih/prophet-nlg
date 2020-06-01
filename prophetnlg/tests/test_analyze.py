import unittest
from prophetnlg.analysis.fin import (
    FinSentenceAnalyzer,
    FinHeuristicSentenceAnalyzer
)


class TestFinAnalysis(unittest.TestCase):
    def setUp(self):
        self.analyzer = FinSentenceAnalyzer()

    def test_analyze_text(self):
        sentences = list(self.analyzer.analyze_text('Kissa hyppäsi katolle.'))
        self.assertEqual(len(sentences), 1)
        self.assertEqual(len(sentences[0].tokens), 4)
        self.assertEqual(
            [t.lemma for t in sentences[0].tokens],
            ['Kissa', 'hypätä', 'katto', '.']
        )
        first = sentences[0].tokens[0]
        self.assertIn('kissa+N+Sg+Nom', first.analyses['uralic'].analysis)
        self.assertEqual(sentences[0].as_text(), 'Kissa hyppäsi katolle.')


class TestFinHeuristicAnalysis(unittest.TestCase):
    def setUp(self):
        self.analyzer = FinHeuristicSentenceAnalyzer()

    def test_analyze_text(self):
        sentences = list(self.analyzer.analyze_text('Kissa hyppäsi katolle.'))
        self.assertEqual(len(sentences), 1)
        self.assertEqual(len(sentences[0].tokens), 4)
        self.assertEqual(
            [t.lemma for t in sentences[0].tokens],
            ['Kissa', 'hypätä', 'katto', '.']
        )
        self.assertEqual(sentences[0].as_text(), 'Kissa hyppäsi katolle.')

    def test_analyze_unknown_nouns(self):
        text = 'Belsebyyb istui. Rokkilazerit juoksivat. Xaczhimalkin miettii.'
        sentences = list(self.analyzer.analyze_text(text))
        self.assertEqual(sentences[0].tokens[0].lemma, 'Belsebyyb')
        self.assertEqual(sentences[1].tokens[0].lemma, 'lazeri')
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
        Lepäämme vasta Morkatissa, luvatussa maassa.
        Zamuelista saatiin hassu kuningas.
        Lähdemme joukolla Mökköön.
        Belielillä on peliä.
        Sain tämän setelin Määmiltä.
        Kerro terveisiä Seepaotille.
        '''
        sentences = list(self.analyzer.analyze_text(text))
        self.assertEqual(sentences[0].tokens[0].lemma, 'Belsebyyb')
        self.assertEqual(sentences[1].tokens[1].lemma, 'turppo')
        self.assertEqual(sentences[2].tokens[0].lemma, 'Xaczhimal')
        self.assertEqual(sentences[3].tokens[0].lemma, 'Mooses')
        self.assertEqual(sentences[4].tokens[2].lemma, 'ismaeli')
        self.assertEqual(sentences[5].tokens[2].lemma, 'katti')
        self.assertEqual(sentences[6].tokens[0].lemma, 'Zamuel')
        self.assertEqual(sentences[7].tokens[2].lemma, 'Mökkö')
        self.assertEqual(sentences[8].tokens[0].lemma, 'Beliel')
        self.assertEqual(sentences[9].tokens[3].lemma, 'määmä')
        self.assertEqual(sentences[10].tokens[2].lemma, 'paotti')


class TestFinHeuristicAnalysisTense(unittest.TestCase):
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
