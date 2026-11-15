import time
from pathlib import Path

import numpy as np
import ollama

from ingestion import ingerer

MODELES = ["bge-m3", "nomic-embed-text", "qwen3-embedding:0.6b", "embeddinggemma"]

# (question, ids des chunks qui contiennent la réponse)
JEU_DE_TEST = [
    ("Combien dure la soutenance ?", ["reglement_pfe.txt#11"]),
    ("Combien coûte une borne de recharge ?", ["faq_solaris.txt#9"]),
    ("Que se passe-t-il si je rends mon rapport en retard ?", ["reglement_pfe.txt#10"]),
    ("Quand ont lieu les soutenances en 2027 ?", ["note_soutenances_scan.pdf#0"]),
    ("Combien touche un boursier qui part au Canada ?", ["guide_stage_etranger.pdf#5"]),
    ("Qui fixe le montant de la gratification ?", ["reglement_pfe.txt#8"]),
    ("Le télétravail depuis le Maroc compte-t-il comme stage à l'étranger ?", ["guide_stage_etranger.pdf#6"]),
    ("Quelle orientation pour mes panneaux ?", ["faq_solaris.txt#4"]),
    ("Kifach kathseb la note dyal PFE ?", ["reglement_pfe.txt#12"]),                       # darija (latin)
    ("Chhal kaykhalles l'entretien dyal les panneaux f l'3am ?", ["faq_solaris.txt#7"]),  # darija (latin)
    ("واش نقدر ندير الستاج ديالي فالخارج؟", ["reglement_pfe.txt#6", "guide_stage_etranger.pdf#0"]),  # darija (arabe)
    ("ما هي مدة ضمان الألواح الشمسية؟", ["faq_solaris.txt#6"]),                            # arabe
    ("What is the warranty on the solar panels?", ["faq_solaris.txt#6"]),                  # anglais
]


def embed(modele, textes):
    vecteurs = np.array(ollama.embed(model=modele, input=textes)["embeddings"])
    return vecteurs / np.linalg.norm(vecteurs, axis=1, keepdims=True)


def evaluer(modele, chunks):
    debut = time.perf_counter()
    vecteurs = embed(modele, [c["texte"] for c in chunks])
    duree = time.perf_counter() - debut
    questions = embed(modele, [q for q, _ in JEU_DE_TEST])

    rangs = []  # position du premier bon chunk (1 = en tête)
    for q, (_, bons) in zip(questions, JEU_DE_TEST):
        classement = [chunks[i]["id"] for i in np.argsort(vecteurs @ q)[::-1]]
        rangs.append(min(classement.index(b) for b in bons) + 1)
    rangs = np.array(rangs)
    return {
        "dimension": vecteurs.shape[1],
        "hit@1": np.mean(rangs <= 1),
        "hit@3": np.mean(rangs <= 3),
        "MRR": np.mean(1 / rangs),
        "temps": duree,
        "rangs": rangs,
    }


if __name__ == "__main__":
    chunks = [c for f in sorted(Path("data").iterdir()) if f.suffix.lower() in (".pdf", ".txt") for c in ingerer(f)]
    disponibles = {m.model for m in ollama.list().models}
    print(f"{len(chunks)} chunks, {len(JEU_DE_TEST)} questions\n")
    print(f"{'Modèle':24} {'dim':>5} {'hit@1':>6} {'hit@3':>6} {'MRR':>5} {'indexation':>11}")
    for modele in MODELES:
        if modele not in disponibles and f"{modele}:latest" not in disponibles:
            print(f"{modele:24} (pas installé : ollama pull {modele})")
            continue
        r = evaluer(modele, chunks)
        print(f"{modele:24} {r['dimension']:5} {r['hit@1']:6.0%} {r['hit@3']:6.0%} {r['MRR']:5.2f} {r['temps']:9.1f} s"
              f"   rangs : {[int(x) for x in r['rangs']]}")