import re
import unicodedata
from collections import defaultdict

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from rag_langchain import base

MODELE_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
NB_CANDIDATS = 20   # combien de chunks chaque méthode propose avant la fusion / le reranking
MOTS_VIDES = set("""
le la les l un une des du de d au aux et ou a à en dans sur pour par avec sans ce cet cette ces se sa son ses
leur leurs il elle ils elles on je tu nous vous qui que quoi qu est sont être faut peut quel quelle quels quelles
combien comment est-ce ne pas plus y mon ma mes ton ta tes the of is what
""".split())

_reranker = None


# ---------- 1. La recherche par mots : BM25 ----------

def tokeniser(texte):
    """'Quelle est la Durée ?' -> ['duree'] : minuscules, sans accents, sans mots vides."""
    texte = unicodedata.normalize("NFKD", texte.lower())
    texte = "".join(c for c in texte if not unicodedata.combining(c))  # é -> e
    return [mot for mot in re.findall(r"\w+", texte) if mot not in MOTS_VIDES]


def tous_les_documents():
    """Relit tous les chunks stockés dans Chroma (module 4)."""
    d = base.get(include=["documents", "metadatas"])
    return [Document(page_content=t, metadata=m, id=i) for i, t, m in zip(d["ids"], d["documents"], d["metadatas"])]


class RechercheBM25:
    def __init__(self, documents):
        self.documents = documents
        self.bm25 = BM25Okapi([tokeniser(d.page_content) for d in documents])

    def chercher(self, question, k=NB_CANDIDATS):
        scores = self.bm25.get_scores(tokeniser(question))
        meilleurs = np.argsort(scores)[::-1][:k]
        return [self.documents[i] for i in meilleurs if scores[i] > 0]  # score 0 = aucun mot en commun


# ---------- 2. La recherche par le sens : dense (bge-m3 + Chroma) ----------

def recherche_dense(question, k=NB_CANDIDATS):
    return base.similarity_search(question, k=k)


# ---------- 3. La fusion des classements : RRF ----------

def fusion_rrf(classements, k=60):
    """Reciprocal Rank Fusion : chaque document gagne 1 / (k + rang) dans chaque classement."""
    scores, documents = defaultdict(float), {}
    for classement in classements:
        for rang, doc in enumerate(classement, start=1):
            scores[doc.id] += 1 / (k + rang)
            documents[doc.id] = doc
    return [documents[i] for i in sorted(scores, key=scores.get, reverse=True)]


# ---------- 4. Le reranking : un cross-encodeur relit chaque paire (question, chunk) ----------

def reranker(question, documents, k):
    global _reranker
    if _reranker is None:  # chargé une seule fois, au premier besoin
        _reranker = CrossEncoder(MODELE_RERANKER)
    scores = _reranker.predict([(question, d.page_content) for d in documents])
    ordre = np.argsort(scores)[::-1][:k]
    return [documents[i] for i in ordre]


# ---------- 5. Tout ensemble ----------

class RechercheHybride:
    def __init__(self):
        self.bm25 = RechercheBM25(tous_les_documents())

    def chercher(self, question, k=4, rerank=True):
        candidats = fusion_rrf([recherche_dense(question), self.bm25.chercher(question)])
        if rerank:
            return reranker(question, candidats[:NB_CANDIDATS], k)
        return candidats[:k]