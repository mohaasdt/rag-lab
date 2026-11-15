import json
import re
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from rag_langchain import formater, indexer_si_vide, llm, prompt
from recherche import RechercheBM25, fusion_rrf, recherche_dense, tous_les_documents

JEU = json.loads(Path("jeu_eval.json").read_text(encoding="utf-8"))
PHRASE_REFUS = "Je ne trouve pas cette information"
TOP_K = 4

generation = prompt | llm | StrOutputParser()


# ---------- Juge 1 : la réponse est-elle correcte ? (comparée à la réponse attendue) ----------

class Correction(BaseModel):
    justification: str = Field(description="Une phrase qui compare la RÉPONSE à la RÉPONSE ATTENDUE")
    verdict: Literal["correct", "partiel", "faux"]


juge_correction = ChatPromptTemplate.from_messages([
    ("system", "Tu es un correcteur strict. Compare la RÉPONSE d'un assistant à la RÉPONSE ATTENDUE.\n"
               "- correct : toutes les informations importantes de la réponse attendue sont présentes, sans erreur.\n"
               "- partiel : ce qui est dit est juste, mais il manque une information importante.\n"
               "- faux : au moins une information est fausse, ou la réponse ne répond pas à la question.\n"
               "Seul le sens compte (la langue et la formulation n'ont pas d'importance). Ignore les numéros entre crochets."),
    ("human", "QUESTION : {question}\nRÉPONSE ATTENDUE : {attendue}\nRÉPONSE : {reponse}"),
]) | llm.with_structured_output(Correction)


# ---------- Juge 2 : la réponse est-elle fidèle au contexte ? (pas d'invention) ----------

class Affirmation(BaseModel):
    texte: str = Field(description="Une information courte tirée de la RÉPONSE")
    soutenue: bool = Field(description="True si le CONTEXTE contient cette information, même avec d'autres mots")


class Fidelite(BaseModel):
    affirmations: list[Affirmation]


juge_fidelite = ChatPromptTemplate.from_messages([
    ("system", "Découpe la RÉPONSE en affirmations courtes (une information par affirmation). "
               "Pour chacune, indique si elle est soutenue par le CONTEXTE. N'utilise pas tes propres connaissances : "
               "seul le CONTEXTE compte. Ignore les numéros entre crochets."),
    ("human", "CONTEXTE :\n{contexte}\n\nRÉPONSE : {reponse}"),
]) | llm.with_structured_output(Fidelite)


# ---------- Évaluer une question ----------

def evaluer_question(item, chercher):
    debut = time.perf_counter()
    documents = chercher(item["question"])
    contexte = formater(documents)
    reponse = generation.invoke({"contexte": contexte, "question": item["question"]})
    latence = time.perf_counter() - debut

    ids = [d.id for d in documents]
    attendus = item["chunks"]
    r = {"question": item["question"], "type": item["type"], "reponse": reponse, "ids": ids, "latence": latence}

    # Recherche : les bons chunks sont-ils dans le top-k ?
    r["rappel_contexte"] = sum(c in ids for c in attendus) / len(attendus) if attendus else None

    # Génération : refus, correction, fidélité, citations
    r["refus"] = PHRASE_REFUS in reponse
    if item["type"] == "piege":
        r["verdict"] = "correct" if r["refus"] else "faux"
        r["justification"] = "refus attendu" + (" : OK" if r["refus"] else " : le modèle a répondu quand même")
    elif r["refus"]:
        r["verdict"], r["justification"] = "faux", "refus injustifié : la réponse était dans les documents"
    else:
        c = juge_correction.invoke({"question": item["question"], "attendue": item["reponse_attendue"], "reponse": reponse})
        r["verdict"], r["justification"] = c.verdict, c.justification

    r["fidelite"] = None
    if not r["refus"]:
        f = juge_fidelite.invoke({"contexte": contexte, "reponse": reponse})
        if f.affirmations:
            r["fidelite"] = sum(a.soutenue for a in f.affirmations) / len(f.affirmations)
            r["non_soutenues"] = [a.texte for a in f.affirmations if not a.soutenue]

    cites = {int(n) for n in re.findall(r"\[(\d+)\]", reponse)}
    r["citation_ok"] = any(1 <= n <= len(ids) and ids[n - 1] in attendus for n in cites) if attendus and not r["refus"] else None
    return r


def resumer(nom, resultats):
    moyenne = lambda cle: np.mean([r[cle] for r in resultats if r[cle] is not None])
    verdicts = [r["verdict"] for r in resultats]
    pieges = [r for r in resultats if r["type"] == "piege"]
    print(f"\n=== {nom} ({len(resultats)} questions) ===")
    print(f"  Rappel du contexte (top-{TOP_K})  : {moyenne('rappel_contexte'):.0%}")
    print(f"  Réponses correctes            : {verdicts.count('correct') / len(verdicts):.0%}"
          f"   (partielles {verdicts.count('partiel') / len(verdicts):.0%}, fausses {verdicts.count('faux') / len(verdicts):.0%})")
    print(f"  Fidélité au contexte          : {moyenne('fidelite'):.0%}")
    print(f"  Citations justes              : {moyenne('citation_ok'):.0%}")
    print(f"  Pièges bien refusés           : {sum(r['refus'] for r in pieges)}/{len(pieges)}")
    print(f"  Refus injustifiés             : {sum(r['refus'] and r['type'] != 'piege' for r in resultats)}")
    print(f"  Latence moyenne               : {moyenne('latence'):.1f} s (recherche + génération)")
    print("  Par type :", ", ".join(
        f"{t} {sum(r['verdict'] == 'correct' for r in resultats if r['type'] == t)}/{sum(r['type'] == t for r in resultats)}"
        for t in dict.fromkeys(r["type"] for r in resultats)))


if __name__ == "__main__":
    indexer_si_vide()
    bm25 = RechercheBM25(tous_les_documents())
    configurations = {
        "dense": lambda q: recherche_dense(q, k=TOP_K),
        "hybride": lambda q: fusion_rrf([recherche_dense(q), bm25.chercher(q)])[:TOP_K],
    }
    choix = sys.argv[1:] or list(configurations)
    for nom in choix:
        resultats = [evaluer_question(item, configurations[nom]) for item in JEU]
        Path(f"resultats_{nom}.json").write_text(json.dumps(resultats, ensure_ascii=False, indent=2), encoding="utf-8")
        resumer(nom, resultats)