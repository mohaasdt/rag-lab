import time

import faiss
import numpy as np

N, D = 50_000, 1024        # 50 000 vecteurs de 1024 nombres (comme bge-m3)
NB_REQUETES, K = 200, 10   # 200 questions, top-10
rng = np.random.default_rng(0)


def normaliser(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def fabriquer(nb, centres):
    """Vecteurs « réalistes » : chacun = un sujet (centre) + du bruit."""
    choix = rng.integers(0, len(centres), nb)
    return normaliser(centres[choix] + 1.5 * rng.standard_normal((nb, D), dtype=np.float32))


def chronometrer(index, requetes):
    """Renvoie les ids trouvés et le temps moyen par requête, en millisecondes."""
    debut = time.perf_counter()
    _, ids = index.search(requetes, K)
    return ids, (time.perf_counter() - debut) / len(requetes) * 1000


def rappel(ids_approx, ids_exacts):
    """Part des vrais 10 plus proches voisins retrouvés par l'index approximatif."""
    return np.mean([len(set(a) & set(e)) / K for a, e in zip(ids_approx, ids_exacts)])


centres = rng.standard_normal((5000, D), dtype=np.float32)  # 5000 « sujets »
base = fabriquer(N, centres)
requetes = fabriquer(NB_REQUETES, centres)
print(f"Base : {N} vecteurs x {D} dimensions = {base.nbytes / 1e6:.0f} Mo\n")

# 1. Recherche exacte (brute force) : la référence
index_exact = faiss.IndexFlatIP(D)          # IP = produit scalaire (= cosinus, vecteurs normalisés)
index_exact.add(base)
ids_exacts, t = chronometrer(index_exact, requetes)
print(f"Exact (Flat)          : {t:6.2f} ms/requête   rappel@10 100.0 %")

# 2. HNSW : un graphe de voisins
debut = time.perf_counter()
index_hnsw = faiss.IndexHNSWFlat(D, 32, faiss.METRIC_INNER_PRODUCT)  # 32 voisins par point
index_hnsw.hnsw.efConstruction = 100
index_hnsw.add(base)
print(f"\nHNSW construit en {time.perf_counter() - debut:.1f} s")
for ef in (16, 64, 256):
    index_hnsw.hnsw.efSearch = ef
    ids, t = chronometrer(index_hnsw, requetes)
    print(f"HNSW efSearch={ef:<4}    : {t:6.2f} ms/requête   rappel@10 {rappel(ids, ids_exacts):6.1%}")

# 3. IVF : des « tiroirs » (clusters)
debut = time.perf_counter()
index_ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(D), D, 256, faiss.METRIC_INNER_PRODUCT)  # 256 tiroirs
index_ivf.train(base)
index_ivf.add(base)
print(f"\nIVF construit en {time.perf_counter() - debut:.1f} s")
for nprobe in (1, 8, 32):
    index_ivf.nprobe = nprobe
    ids, t = chronometrer(index_ivf, requetes)
    print(f"IVF nprobe={nprobe:<4}       : {t:6.2f} ms/requête   rappel@10 {rappel(ids, ids_exacts):6.1%}")