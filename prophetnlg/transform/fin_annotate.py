from prophetnlg import Sentence
from .annotate import WordAnnotationTransform


class FinWordAnnotationTransform(WordAnnotationTransform):
    def annotate_pos(self, sentence: Sentence) -> Sentence:
        return self._annotate(
            sentence,
            lambda t: 'NEG' if t.lemma == 'ei' else t.pos
        )
