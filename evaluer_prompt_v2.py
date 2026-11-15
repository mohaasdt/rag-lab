import json
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import evaluer_rag as E
from rag_langchain import indexer_si_vide, llm
from recherche import recherche_dense

SYSTEME_V2 = """Tu es un assistant qui répond aux questions à partir de documents.
Règles :
- Utilise UNIQUEMENT les informations du CONTEXTE.
- Réponds de façon COMPLÈTE : donne toutes les informations utiles du CONTEXTE (chiffres, conditions, exceptions).
- Si plusieurs passages sont utiles, combine-les et cite chacun.
- Si le CONTEXTE ne contient qu'une partie de la réponse, donne cette partie et dis clairement ce qui manque.
- Si le CONTEXTE ne contient RIEN d'utile pour la question, réponds exactement : "Je ne trouve pas cette information dans les documents."
- Après chaque information, cite sa source avec son numéro entre crochets, par exemple [1].
- Réponds TOUJOURS dans la langue de la question (français, anglais, arabe ou darija)."""

prompt_v2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEME_V2),
    ("human", "CONTEXTE :\n{contexte}\n\nQUESTION : {question}"),
])

if __name__ == "__main__":
    indexer_si_vide()
    E.generation = prompt_v2 | llm | StrOutputParser()  # on ne change QUE le prompt
    resultats = [E.evaluer_question(item, lambda q: recherche_dense(q, k=E.TOP_K)) for item in E.JEU]
    Path("resultats_dense_prompt_v2.json").write_text(json.dumps(resultats, ensure_ascii=False, indent=2), encoding="utf-8")
    E.resumer("dense + prompt v2", resultats)