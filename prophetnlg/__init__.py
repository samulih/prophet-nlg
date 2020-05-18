from typing import Any, Dict, List, NamedTuple


class WordAnalysis(NamedTuple):
    text: str
    analysis: Any


class SentenceToken(NamedTuple):
    text: str
    lemma: str = ''
    pos: str = ''
    lang: str = ''
    morphology: str = ''
    prefix_morphology: str = ''
    analyses: Dict[str, WordAnalysis] = {}
    category: str = ''
    spaces_after: str = ' '
    cap: bool = False

    def with_analysis(self, analysis: Any, analysis_type: str):
        analyses = dict(
            self.analyses, **{
                analysis_type: WordAnalysis(text=self.text, analysis=analysis)
            }
        )
        return self._replace(analyses=analyses)


class Sentence(NamedTuple):
    tokens: List[SentenceToken]

    def as_text(self):
        return ''.join(
            f'{t.text.capitalize() if t.cap else t.text}{t.spaces_after}'
            for t in self.tokens
        )
