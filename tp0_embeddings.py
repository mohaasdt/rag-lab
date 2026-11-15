import numpy as np
import ollama

phrases = [
    "Comment s'inscrire au master en France ?",
    "Quelle est la procédure de candidature pour un M2 ?",
    "Kifach ndir l'inscription f master f França ?",  # darija
    "La recette du tajine aux pruneaux",
]

resultat = ollama.embed(model="bge-m3", input=phrases)
vecteurs = np.array(resultat["embeddings"])

print("Forme :", vecteurs.shape)          # (4, 1024) : 4 phrases, 1024 nombres chacune
print("Début du 1er vecteur :", vecteurs[0][:5])


def cosinus(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


for i in range(len(phrases)):
    for j in range(i + 1, len(phrases)):
        print(f"{cosinus(vecteurs[i], vecteurs[j]):.3f}  |  {phrases[i]}  <->  {phrases[j]}")