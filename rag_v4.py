import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from rag_langchain import decrire_source, formater, indexer_si_vide, llm, prompt
from recherche import RechercheHybride

indexer_si_vide()
recherche = RechercheHybride()  # construit l'index BM25 à partir des chunks de Chroma
RERANK = "--sans-reranker" not in sys.argv

# Seule la case CHERCHER change : le retriever Chroma est remplacé par la recherche hybride
retriever = RunnableLambda(lambda question: recherche.chercher(question, k=4, rerank=RERANK))

chaine = (
    RunnableParallel(documents=retriever, question=RunnablePassthrough())
    .assign(contexte=lambda x: formater(x["documents"]))
    .assign(reponse=prompt | llm | StrOutputParser())
)

if __name__ == "__main__":
    print(f"Recherche hybride (BM25 + bge-m3, fusion RRF){' + reranker' if RERANK else ''}")
    while True:
        question = input("\nQuestion (q pour quitter) : ").strip()
        if question.lower() == "q":
            break
        if not question:
            continue
        documents = []
        for morceau in chaine.stream(question):
            if "documents" in morceau:
                documents = morceau["documents"]
            if "reponse" in morceau:
                print(morceau["reponse"], end="", flush=True)
        print("\n\nSources :")
        for n, d in enumerate(documents, start=1):
            print(f"  [{n}] {decrire_source(d.metadata)}  ({d.id})")