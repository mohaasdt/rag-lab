import json
import sys
from pathlib import Path

import numpy as np
import ollama

MODELE_EMBEDDING = "bge-m3"
MODELE_LLM = "qwen2.5:7b"
DOSSIER_DATA = Path("data")
FICHIER_VECTEURS = Path("index/vecteurs.npy")
FICHIER_CHUNKS = Path("index/chunks.json")
TAILLE_CHUNK = 600   # en caractères
TOP_K = 4

SYSTEME = """Tu es un assistant qui répond aux questions à partir de documents.
Règles :
- Utilise UNIQUEMENT les informations du CONTEXTE.
- Si la réponse n'est pas dans le CONTEXTE, réponds exactement : "Je ne trouve pas cette information dans les documents."
- Après chaque information, cite sa source avec son numéro entre crochets, par exemple [1].
- Réponds dans la langue de la question, de façon courte et précise."""


# ---------- Phase A : indexation ----------

def charger_documents(dossier):
    """1. Charger : lit tous les fichiers .txt du dossier."""
    return [
        {"source": chemin.name, "texte": chemin.read_text(encoding="utf-8")}
        for chemin in sorted(dossier.glob("*.txt"))
    ]


def decouper(texte, taille_max=TAILLE_CHUNK):
    """2. Découper : regroupe les paragraphes jusqu'à taille_max caractères."""
    paragraphes = [p.strip() for p in texte.split("\n\n") if p.strip()]
    chunks, courant = [], ""
    for p in paragraphes:
        if courant and len(courant) + len(p) > taille_max:
            chunks.append(courant)
            courant = p
        else:
            courant = f"{courant}\n\n{p}" if courant else p
    if courant:
        chunks.append(courant)
    return chunks


def embed(textes):
    """3. Embedding : textes -> vecteurs normalisés (longueur 1)."""
    vecteurs = np.array(ollama.embed(model=MODELE_EMBEDDING, input=textes)["embeddings"])
    return vecteurs / np.linalg.norm(vecteurs, axis=1, keepdims=True)


def construire_index():
    """4. Stocker : sauvegarde les chunks (texte + métadonnées) et leurs vecteurs."""
    chunks = []
    for doc in charger_documents(DOSSIER_DATA):
        for i, texte in enumerate(decouper(doc["texte"])):
            chunks.append({"id": f"{doc['source']}#{i}", "source": doc["source"], "texte": texte})
    vecteurs = embed([c["texte"] for c in chunks])

    FICHIER_VECTEURS.parent.mkdir(exist_ok=True)
    np.save(FICHIER_VECTEURS, vecteurs)
    FICHIER_CHUNKS.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Index construit : {len(chunks)} chunks.")
    return chunks, vecteurs


def charger_index():
    """Recharge l'index s'il existe déjà, sinon le construit."""
    if FICHIER_VECTEURS.exists() and FICHIER_CHUNKS.exists():
        return json.loads(FICHIER_CHUNKS.read_text(encoding="utf-8")), np.load(FICHIER_VECTEURS)
    return construire_index()


# ---------- Phase B : question ----------

def chercher(question, chunks, vecteurs, k=TOP_K):
    """Retriever : renvoie les k chunks les plus proches de la question."""
    scores = vecteurs @ embed([question])[0]
    meilleurs = np.argsort(scores)[::-1][:k]
    return [{**chunks[i], "score": float(scores[i])} for i in meilleurs]


def construire_contexte(resultats):
    """Numérote les chunks pour que le LLM puisse les citer : [1], [2]..."""
    return "\n\n".join(
        f"[{n}] (source : {r['source']})\n{r['texte']}" for n, r in enumerate(resultats, start=1)
    )


def repondre(question, chunks, vecteurs, debug=False):
    resultats = chercher(question, chunks, vecteurs)

    if debug:
        print("\n--- Chunks récupérés ---")
        for n, r in enumerate(resultats, start=1):
            print(f"[{n}] {r['score']:.3f}  {r['id']}  {r['texte'][:80]!r}")
        print("------------------------\n")

    messages = [
        {"role": "system", "content": SYSTEME},
        {"role": "user", "content": f"CONTEXTE :\n{construire_contexte(resultats)}\n\nQUESTION : {question}"},
    ]
    flux = ollama.chat(
        model=MODELE_LLM,
        messages=messages,
        options={"temperature": 0, "num_ctx": 8192},
        stream=True,
    )
    for morceau in flux:
        print(morceau["message"]["content"], end="", flush=True)

    print("\n\nSources :")
    for n, r in enumerate(resultats, start=1):
        print(f"  [{n}] {r['id']}")


if __name__ == "__main__":
    debug = "--debug" in sys.argv
    chunks, vecteurs = charger_index()
    while True:  
        question = input("\nQuestion (q pour quitter) : ").strip()
        if question.lower() == "q":
            break
        if question:
            repondre(question, chunks, vecteurs, debug=debug)