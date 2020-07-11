from pydantic import BaseModel


class SentenceTokenGeneratorBase(BaseModel):
    language: str = 'finnish'
    lang: str = 'fin'

