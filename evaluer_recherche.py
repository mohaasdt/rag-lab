import time

import numpy as np

from rag_langchain import indexer_si_vide
from recherche import NB_CANDIDATS, RechercheBM25, fusion_rrf, recherche_dense, reranker, tous_les_documents

# (question, ids des chunks qui contiennent la réponse)
JEU_DE_TEST = [
    # mots exacts, numéros, codes
    ("Que dit l'article 9 ?", ["reglement_pfe.txt#9"]),
    ("article 13", ["reglement_pfe.txt#13"]),
    ("salle B12", ["note_soutenances_scan.pdf#0"]),
    ("porte 4", ["guide_stage_etranger.pdf#9"]),
    ("borne 7,4 kW", ["faq_solaris.txt#9"]),
    ("Combien d'exemplaires de la convention ?", ["reglement_pfe.txt#5"]),
    ("ensoleillement Tanger", ["faq_solaris.txt#3"]),
    ("date limite demande allocation", ["guide_stage_etranger.pdf#5"]),
    # synonymes et reformulations
    ("Quelle est la rémunération minimale conseillée pour un stagiaire ?", ["reglement_pfe.txt#8"]),
    ("Le rapport peut-il rester secret ?", ["reglement_pfe.txt#14"]),
    ("Combien de temps faut-il pour que le responsable accepte mon sujet ?", ["reglement_pfe.txt#4"]),
    ("Mon onduleur est garanti combien de temps ?", ["faq_solaris.txt#6"]),
    ("Que faire en cas de panne urgente ?", ["faq_solaris.txt#12"]),
    ("Puis-je soutenir à distance si je suis encore à l'étranger ?", ["guide_stage_etranger.pdf#7"]),
    ("Combien de jours d'absence avant que le stage soit annulé ?", ["reglement_pfe.txt#9"]),
    ("Quel montant l'école verse-t-elle pour un stage en Allemagne ?", ["guide_stage_etranger.pdf#5"]),
    # darija, arabe, anglais
    ("Kifach kathseb la note dyal PFE ?", ["reglement_pfe.txt#12"]),
    ("واش نقدر ندير الستاج ديالي فالخارج؟", ["reglement_pfe.txt#6", "guide_stage_etranger.pdf#0"]),
    ("ما هي مدة ضمان الألواح الشمسية؟", ["faq_solaris.txt#6"]),
    ("How long does the thesis defense last?", ["reglement_pfe.txt#11"]),
]


def rang(resultats, bons):
    """Position (à partir de 1) du premier bon chunk, ou None s'il n'est pas dans les résultats."""
    ids = [d.id for d in resultats]
    positions = [ids.index(b) + 1 for b in bons if b in ids]
    return min(positions) if positions else None


def evaluer(nom, methode):
    debut = time.perf_counter()
    rangs = [rang(methode(q), bons) for q, bons in JEU_DE_TEST]
    duree = (time.perf_counter() - debut) / len(JEU_DE_TEST) * 1000
    hit = lambda k: np.mean([r is not None and r <= k for r in rangs])
    mrr = np.mean([1 / r if r else 0 for r in rangs])
    affichage = " ".join("-" if r is None else str(r) for r in rangs)
    print(f"{nom:24} hit@1 {hit(1):4.0%}  hit@4 {hit(4):4.0%}  MRR {mrr:.2f}  {duree:5.0f} ms/question   rangs : {affichage}")


if __name__ == "__main__":
    indexer_si_vide()
    bm25 = RechercheBM25(tous_les_documents())
    hybride = lambda q: fusion_rrf([recherche_dense(q), bm25.chercher(q)])
    reranker(JEU_DE_TEST[0][0], tous_les_documents()[:2], 1)  # charge le modèle avant de chronométrer

    print(f"{len(JEU_DE_TEST)} questions\n")
    evaluer("Dense (bge-m3)", recherche_dense)
    evaluer("BM25", bm25.chercher)
    evaluer("Hybride (RRF)", hybride)
    evaluer("Dense + reranker", lambda q: reranker(q, recherche_dense(q), NB_CANDIDATS))
    evaluer("Hybride + reranker", lambda q: reranker(q, hybride(q)[:NB_CANDIDATS], NB_CANDIDATS))