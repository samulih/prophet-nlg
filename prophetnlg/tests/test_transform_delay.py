from more_itertools import take
from typing import Iterator, List, Mapping
import unittest
from prophetnlg import Sentence
from prophetnlg.analysis.fin import FinHeuristicSentenceAnalyzer
from prophetnlg.transform.delay import SentenceDelayTransform


class TestSentenceDelayTransform(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.analyzer = FinHeuristicSentenceAnalyzer()

    def test_delay(self):
        transform = SentenceDelayTransform(buffer_size=2)
        sentences = list(self.analyzer.analyze_text('Eka lause tää on. Tämä on toka. Kolmatta viedään. Neljäskin löytyi.'))
        delayed = list(transform.transform_stream(sentences))
        self.assertEqual(delayed[0:2], [Sentence(), Sentence()])
        self.assertEqual(delayed[2:], sentences)

    def test_delay_loop(self):
        transform = SentenceDelayTransform(buffer_size=2, looping=True)
        sentences = list(self.analyzer.analyze_text('Eka lause tää on. Tämä on toka. Kolmatta viedään. Neljäskin löytyi.'))
        delayed = list(take(10, transform.transform_stream(sentences)))
        self.assertEqual(delayed[:2], [Sentence(), Sentence()])
        self.assertEqual(delayed[2:6], sentences)
        self.assertEqual(delayed[6:8], sentences[-2:])
        self.assertEqual(delayed[8:10], sentences[-2:])

    def test_delay_loop_change_state(self):
        transform = SentenceDelayTransform(buffer_size=2, looping=True)
        sentences = list(self.analyzer.analyze_text('Eka lause tää on. Tämä on toka. Kolmatta viedään. Neljäskin löytyi.'))
        sentences2 = list(self.analyzer.analyze_text('Viides lause menossa. Kuudes tulossa. Seitsemäs päättää.'))
        delayed = list(take(3, transform.transform_stream(sentences)))
        delayed2 = list(take(7, transform.transform_stream(sentences2)))
        delayed = delayed + delayed2
        self.assertEqual(delayed[:2], [Sentence(), Sentence()])
        self.assertEqual(delayed[2:5], sentences[:3])
        self.assertEqual(delayed[5:8], sentences2)
        self.assertEqual(delayed[8:], sentences2[-2:])
