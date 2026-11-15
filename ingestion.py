import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf
from rapidocr_onnxruntime import RapidOCR

DOSSIER_DATA = Path("data")
FICHIER_SORTIE = Path("index/chunks_ingestion.json")
TAILLE_CHUNK = 800        # en caractères
CHEVAUCHEMENT_MAX = 250   # on répète le dernier morceau du chunk précédent s'il est plus court que ça
SEUIL_SCAN = 30           # moins de 30 caractères de texte sur une page = page scannée
RE_TITRE = re.compile(r"^(\d+(\.\d+)*\.?|Article\s+\d+)\s+\S")
RE_NUMERO_PAGE = re.compile(r"^page\s+\d+(\s*/\s*\d+)?$", re.IGNORECASE)

_moteur_ocr = None


# ---------- 1. Extraire : fichier -> paragraphes {texte, pages, methode} ----------

def ocr(page):
    """Lit le texte d'une page scannée : page -> image -> lignes de texte."""
    global _moteur_ocr
    if _moteur_ocr is None:  # le modèle d'OCR est chargé une seule fois, au premier besoin
        _moteur_ocr = RapidOCR()
    image = page.get_pixmap(dpi=200).tobytes("png")
    resultat, _ = _moteur_ocr(image)
    return [ligne[1] for ligne in resultat or []]  # chaque ligne = [position, texte, confiance]


def extraire_pdf(chemin):
    paragraphes = []
    with pymupdf.open(chemin) as pdf:
        nb_pages = len(pdf)
        for page in pdf:
            if len(page.get_text().strip()) < SEUIL_SCAN:
                textes, methode = ocr(page), "ocr"
            else:
                blocs = page.get_text("blocks", sort=True)
                textes, methode = [b[4] for b in blocs if b[6] == 0], "texte"  # b[6] == 0 : bloc de texte
            for texte in textes:
                paragraphes.append({"texte": texte, "pages": [page.number + 1], "methode": methode})
    return paragraphes, nb_pages


def extraire_txt(chemin):
    lignes = chemin.read_text(encoding="utf-8").splitlines()
    return [{"texte": l, "pages": [], "methode": "texte"} for l in lignes], 1


# ---------- 2. Nettoyer ----------

def nettoyer_texte(texte):
    texte = unicodedata.normalize("NFKC", texte)          # ligatures : "ﬁ" -> "fi"
    texte = re.sub(r"(\w)-\n(\w)", r"\1\2", texte)        # césure en fin de ligne : "infor-\nmation"
    return " ".join(texte.split())                         # retours à la ligne et espaces multiples


def nettoyer(paragraphes, nb_pages):
    """Nettoie chaque texte, puis enlève les numéros de page et les en-têtes/pieds de page répétés."""
    for p in paragraphes:
        p["texte"] = nettoyer_texte(p["texte"])
    paragraphes = [p for p in paragraphes if p["texte"]]
    compte = Counter(p["texte"] for p in paragraphes)
    return [
        p for p in paragraphes
        if not RE_NUMERO_PAGE.match(p["texte"])
        and not (nb_pages > 1 and compte[p["texte"]] >= max(2, nb_pages / 2))
    ]


def recoller_coupures(paragraphes):
    """Recolle un paragraphe coupé en deux (changement de page, ligne d'OCR) :
    le précédent ne finit pas par une ponctuation forte et la suite commence par une minuscule."""
    resultat = []
    for p in paragraphes:
        precedent = resultat[-1] if resultat else None
        if precedent and not precedent["texte"].endswith((".", ":", "!", "?")) and p["texte"][0].islower():
            precedent["texte"] += " " + p["texte"]
            precedent["pages"] = sorted(set(precedent["pages"] + p["pages"]))
        else:
            resultat.append(dict(p))
    return resultat


# ---------- 3. Structurer : retrouver les titres ----------

def est_titre(texte):
    return len(texte) < 100 and (RE_TITRE.match(texte) is not None or texte.endswith("?"))


def ajouter_sections(paragraphes):
    section = ""
    for p in paragraphes:
        p["titre"] = est_titre(p["texte"])
        if p["titre"]:
            section = p["texte"]
        p["section"] = section
    return paragraphes


# ---------- 4. Découper en chunks ----------

def couper_aux_phrases(texte, taille_max):
    """Coupe un paragraphe trop long aux fins de phrases."""
    morceaux, courant = [], ""
    for phrase in re.split(r"(?<=[.!?])\s+", texte):
        if courant and len(courant) + len(phrase) + 1 > taille_max:
            morceaux.append(courant)
            courant = phrase
        else:
            courant = f"{courant} {phrase}".strip()
    if courant:
        morceaux.append(courant)
    return morceaux


def creer_chunk(morceaux, source, numero):
    section = morceaux[0]["section"]
    corps = "\n\n".join(m["texte"] for m in morceaux)
    texte = corps if not section or corps.startswith(section) else f"{section}\n\n{corps}"  # titre en tête
    return {
        "id": f"{source}#{numero}",
        "source": source,
        "pages": sorted({page for m in morceaux for page in m["pages"]}),
        "section": section,
        "methode": morceaux[0]["methode"],
        "texte": texte,
    }


def decouper(paragraphes, source, taille_max=TAILLE_CHUNK):
    """Un chunk ne mélange jamais deux sections. Paragraphes regroupés jusqu'à taille_max, avec chevauchement."""
    morceaux = [{**p, "texte": t} for p in paragraphes for t in couper_aux_phrases(p["texte"], taille_max)]
    groupes, courant = [], []
    for m in morceaux:
        nouvelle_section = courant and m["section"] != courant[-1]["section"]
        trop_long = courant and sum(len(x["texte"]) for x in courant) + len(m["texte"]) > taille_max
        if nouvelle_section or trop_long:
            groupes.append(courant)
            dernier = courant[-1]
            garder = (trop_long and not nouvelle_section and not dernier["titre"]
                      and len(dernier["texte"]) <= CHEVAUCHEMENT_MAX
                      and len(dernier["texte"]) + len(m["texte"]) <= taille_max)  # le chevauchement doit tenir dans le chunk
            courant = [dernier] if garder else []
        courant.append(m)
    if courant:
        groupes.append(courant)
    groupes = [g for g in groupes if not all(m["titre"] for m in g)]  # pas de chunk fait seulement de titres
    return [creer_chunk(g, source, i) for i, g in enumerate(groupes)]


# ---------- Le pipeline complet ----------

def ingerer(chemin):
    if chemin.suffix.lower() == ".pdf":
        paragraphes, nb_pages = extraire_pdf(chemin)
    else:
        paragraphes, nb_pages = extraire_txt(chemin)
    paragraphes = nettoyer(paragraphes, nb_pages)
    paragraphes = recoller_coupures(paragraphes)
    paragraphes = ajouter_sections(paragraphes)
    return decouper(paragraphes, chemin.name)


if __name__ == "__main__":
    tous = []
    for chemin in sorted(DOSSIER_DATA.iterdir()):
        if chemin.suffix.lower() in (".pdf", ".txt"):
            chunks = ingerer(chemin)
            tous += chunks
            tailles = [len(c["texte"]) for c in chunks]
            print(f"{chemin.name:30} {len(chunks):3} chunks  (taille min {min(tailles)}, max {max(tailles)})")

    FICHIER_SORTIE.parent.mkdir(exist_ok=True)
    FICHIER_SORTIE.write_text(json.dumps(tous, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(tous)} chunks enregistrés dans {FICHIER_SORTIE}")

    if "--voir" in sys.argv:
        for c in tous:
            print(f"\n=== {c['id']} | pages {c['pages']} | {c['methode']} | section : {c['section']!r}")
            print(c["texte"])