from prophetnlg import Sentence, SentenceToken
from typing import Callable, Mapping, MutableMapping, Sequence, Union


class WordAnnotationTransform:
    def _annotate(self, sentence: Sentence, category_func: Callable) -> Sentence:
        return sentence.replace(tokens=[
                token.replace(category=category_func(token))
                for token in sentence.tokens
            ]
        )

    def annotate_pos(self, sentence: Sentence) -> Sentence:
        return self._annotate(sentence, lambda t: t.pos)
