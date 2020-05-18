from prophetnlg import Sentence, SentenceToken
from typing import Callable, Mapping, MutableMapping, Sequence, Union


class WordReplaceTransform:
    def __init__(self, generator):
        self.generator = generator

    def replace_in_category(
        self,
        sentence: Sentence,
        category_replacements: MutableMapping[str, Sequence[SentenceToken]]
    ) -> Sentence:
        new_tokens = []
        for token in sentence.tokens:
            category = token.category
            if category_replacements.get(category):
                replacement = category_replacements[category].pop(0)
                new_token = self.generator.token_with_new_lemma(token, replacement)
                new_tokens.append(new_token)
            else:
                new_tokens.append(token)
        return sentence._replace(tokens=new_tokens)

    def replace_lemma(
        self,
        sentence: Sentence,
        lemma_mapping: Mapping[str, str]
    ) -> Sentence:
        new_tokens = []
        for token in sentence.tokens:
            new_lemma = lemma_mapping.get(token.root)
            if new_lemma is not None:
                new_tokens.append(
                    self.generator.token_with_new_lemma(token, new_lemma)
                )
            else:
                new_tokens.append(token)
        return sentence._replace(tokens=new_tokens)
