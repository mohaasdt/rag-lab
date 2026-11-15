import sys
from pathlib import Path

import chromadb
import numpy as np
import ollama

from ingestion import ingerer

MODELE_EMBEDDING = "bge-m3"
MODELE_LLM = "qwen2.5:7b"
DOSSIER_DATA = Path("data")
DOSSIER_CHROMA = "index/chroma"
COLLECTION = "rag_lab"
TOP_K = 4

SYSTEME = """Tu es un assistant qui répond aux questions à partir de documents.
Règles :
- Utilise UNIQUEMENT les informations du CONTEXTE.
- Si la réponse n'est pas dans le CONTEXTE, réponds exactement : "Je ne trouve pas cette information dans les documents."
- Après chaque information, cite sa source avec son numéro entre crochets, par exemple [1].
- Réponds dans la langue de la question, de façon courte et précise."""


def embed(textes):
    """Textes -> liste de vecteurs normalisés (Chroma attend des listes Python)."""
    vecteurs = np.array(ollama.embed(model=MODELE_EMBEDDING, input=textes)["embeddings"])
    return (vecteurs / np.linalg.norm(vecteurs, axis=1, keepdims=True)).tolist()


# ---------- La base vectorielle ----------

def ouvrir_collection():
    client = chromadb.PersistentClient(path=DOSSIER_CHROMA)  # base enregistrée sur le disque
    return client.get_or_create_collection(
        name=COLLECTION,
        configuration={"hnsw": {"space": "cosine"}},  # distance cosinus
        embedding_function=None,                      # on fournit nous-mêmes les vecteurs (bge-m3)
    )


def metadonnees(chunk, date_fichier):
    m = {"source": chunk["source"], "section": chunk["section"], "methode": chunk["methode"], "date_fichier": date_fichier}
    if chunk["pages"]:  # Chroma refuse les listes vides
        m["pages"] = chunk["pages"]
    return m


def indexer_fichier(collection, chemin):
    """(Ré)indexe un fichier : supprime ses anciens chunks, puis ajoute les nouveaux."""
    collection.delete(where={"source": chemin.name})
    chunks = ingerer(chemin)
    date_fichier = chemin.stat().st_mtime
    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=embed([c["texte"] for c in chunks]),
        documents=[c["texte"] for c in chunks],
        metadatas=[metadonnees(c, date_fichier) for c in chunks],
    )
    print(f"  indexé : {chemin.name} ({len(chunks)} chunks)")


def synchroniser(collection):
    """Indexation incrémentale : ne traite que les fichiers nouveaux, modifiés ou supprimés."""
    deja = {}  # {source: date du fichier au moment de l'indexation}
    for m in collection.get(include=["metadatas"])["metadatas"]:
        deja[m["source"]] = m["date_fichier"]

    fichiers = [c for c in sorted(DOSSIER_DATA.iterdir()) if c.suffix.lower() in (".pdf", ".txt")]
    for chemin in fichiers:
        if deja.get(chemin.name) != chemin.stat().st_mtime:  # nouveau ou modifié
            indexer_fichier(collection, chemin)

    for source in set(deja) - {c.name for c in fichiers}:  # supprimé du dossier
        collection.delete(where={"source": source})
        print(f"  supprimé : {source}")
    print(f"Base à jour : {collection.count()} chunks.")


def chercher(collection, question, k=TOP_K, filtre=None):
    r = collection.query(
        query_embeddings=embed([question]),
        n_results=k,
        where=filtre,  # ex. {"source": "reglement_pfe.txt"}
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"id": i, "texte": doc, **meta, "score": 1 - dist}  # distance cosinus = 1 - similarité
        for i, doc, meta, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])
    ]


# ---------- La génération (comme rag_v2) ----------

def decrire_source(r):
    """Ex. : "guide_stage_etranger.pdf, pages 1-2" ou "reglement_pfe.txt"."""
    pages = r.get("pages")
    if not pages:
        return r["source"]
    if len(pages) == 1:
        return f"{r['source']}, page {pages[0]}"
    return f"{r['source']}, pages {pages[0]}-{pages[-1]}"


def construire_contexte(resultats):
    return "\n\n".join(
        f"[{n}] (source : {decrire_source(r)})\n{r['texte']}" for n, r in enumerate(resultats, start=1)
    )


def repondre(collection, question, filtre=None, debug=False):
    resultats = chercher(collection, question, filtre=filtre)
    if not resultats:
        print("Aucun document ne correspond à ce filtre.")
        return

    if debug:
        print("\n--- Chunks récupérés ---")
        for n, r in enumerate(resultats, start=1):
            print(f"[{n}] {r['score']:.3f}  {r['id']}  {r['texte'][:80]!r}")
        print("------------------------\n")

    messages = [
        {"role": "system", "content": SYSTEME},
        {"role": "user", "content": f"CONTEXTE :\n{construire_contexte(resultats)}\n\nQUESTION : {question}"},
    ]
    flux = ollama.chat(model=MODELE_LLM, messages=messages, options={"temperature": 0, "num_ctx": 8192}, stream=True)
    for morceau in flux:
        print(morceau["message"]["content"], end="", flush=True)

    print("\n\nSources :")
    for n, r in enumerate(resultats, start=1):
        print(f"  [{n}] {decrire_source(r)}  ({r['id']})")


def lire_filtre(saisie):
    """'@reglement_pfe.txt Combien dure… ?' -> ({"source": "reglement_pfe.txt"}, "Combien dure… ?")"""
    if saisie.startswith("@") and " " in saisie:
        source, question = saisie[1:].split(" ", 1)
        return {"source": source}, question
    return None, saisie


if __name__ == "__main__":
    debug = "--debug" in sys.argv
    collection = ouvrir_collection()
    synchroniser(collection)
    while True:
        saisie = input("\nQuestion (q pour quitter, @fichier pour filtrer) : ").strip()
        if saisie.lower() == "q":
            break
        if saisie:
            filtre, question = lire_filtre(saisie)
            repondre(collection, question, filtre=filtre, debug=debug)