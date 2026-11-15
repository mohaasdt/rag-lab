from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from rag_langchain import base, llm
from recherche import fusion_rrf


# ---------- Multi-query : plusieurs façons de poser la même question ----------

class Variantes(BaseModel):
    questions: list[str] = Field(description="3 reformulations différentes de la question, en français")


multi_query = ChatPromptTemplate.from_messages([
    ("system", "Tu aides un moteur de recherche documentaire. Écris 3 reformulations différentes de la question "
               "de l'utilisateur, en français standard, avec des synonymes et des mots qu'on trouverait dans un "
               "règlement ou une FAQ. Ne réponds pas à la question."),
    ("human", "{question}"),
]) | llm.with_structured_output(Variantes)


def recherche_multi_query(question, k=20):
    variantes = [question] + multi_query.invoke({"question": question}).questions
    return fusion_rrf([base.similarity_search(v, k=k) for v in variantes]), variantes


# ---------- HyDE : chercher avec une réponse imaginée ----------

hyde = ChatPromptTemplate.from_messages([
    ("system", "Écris un court paragraphe (3 phrases) qui pourrait répondre à la question, dans le style d'un "
               "règlement d'école ou d'une FAQ d'entreprise, en français. Invente les détails si nécessaire : "
               "ce texte sert seulement à la recherche."),
    ("human", "{question}"),
]) | llm | StrOutputParser()


def recherche_hyde(question, k=20):
    document_imagine = hyde.invoke({"question": question})
    return base.similarity_search(document_imagine, k=k), document_imagine