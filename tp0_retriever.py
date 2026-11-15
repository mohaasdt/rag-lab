import numpy as np
import ollama

MODELE_EMBEDDING = "bge-m3"


def embed(textes):
    """Transforme une liste de textes en une matrice (nb_textes, 1024)."""
    resultat = ollama.embed(model=MODELE_EMBEDDING, input=textes)
    return np.array(resultat["embeddings"])


def normaliser(matrice):
    """Met chaque vecteur à une longueur de 1 : le cosinus devient un simple produit scalaire."""
    return matrice / np.linalg.norm(matrice, axis=1, keepdims=True)


# --- Phase A : indexation (faite UNE seule fois) ---
def indexer(phrases):
    return normaliser(embed(phrases))


# --- Phase B : recherche (faite à CHAQUE question) ---
def chercher(question, phrases, index, k=3):
    q = normaliser(embed([question]))[0]   # vecteur de la question, longueur 1
    scores = index @ q                     # cosinus avec TOUTES les phrases d'un coup
    meilleurs = np.argsort(scores)[::-1][:k]  # indices triés du plus grand au plus petit
    return [(phrases[i], float(scores[i])) for i in meilleurs]


if __name__ == "__main__":
    phrases = [
        "Le stage PFE dure six mois et commence en février.",
        "La soutenance a lieu devant un jury de trois enseignants.",
        "Campus France demande un projet d'études motivé pour chaque formation.",
        "Le visa étudiant pour la France nécessite une attestation de ressources.",
        "Un panneau solaire produit plus d'électricité quand il est orienté vers le sud.",
        "Une borne de recharge pour voiture électrique peut être installée dans un parking.",
        "Le couscous est traditionnellement préparé le vendredi au Maroc.",
        "Le tajine aux pruneaux est un plat sucré-salé.",
        "Docker permet de lancer une application dans un conteneur isolé.",
        "PostgreSQL avec l'extension pgvector peut stocker des embeddings.",
    ]

    index = indexer(phrases)  # une seule fois

    questions = [
        "Combien de temps dure le stage de fin d'études ?",
        "Chno khassni bach nakhod visa l França ?",     # darija
        "Où mettre une base vectorielle ?",
        "Quel plat manger le vendredi ?",
    ]

    for question in questions:
        print(f"\nQUESTION : {question}")
        for phrase, score in chercher(question, phrases, index, k=3):
            print(f"  {score:.3f}  {phrase}")