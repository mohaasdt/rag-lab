import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings

from ingestion import ingerer

DOSSIER_DATA = Path("data")
TOP_K = 4

SYSTEME = """Tu es un assistant qui répond aux questions à partir de documents.
Règles :
- Utilise UNIQUEMENT les informations du CONTEXTE.
- Si la réponse n'est pas dans le CONTEXTE, réponds exactement : "Je ne trouve pas cette information dans les documents."
- Après chaque information, cite sa source avec son numéro entre crochets, par exemple [1].
- Réponds dans la langue de la question, de façon courte et précise."""

# ---------- Les composants ----------
embeddings = OllamaEmbeddings(model="bge-m3")
llm = ChatOllama(model="qwen2.5:7b", temperature=0, num_ctx=8192)
base = Chroma(
    collection_name="rag_langchain",
    embedding_function=embeddings,          # LangChain calcule lui-même les embeddings
    persist_directory="index/chroma_langchain",
    collection_configuration={"hnsw": {"space": "cosine"}},
)
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEME),
    ("human", "CONTEXTE :\n{contexte}\n\nQUESTION : {question}"),
])


# ---------- Indexation ----------
def vers_documents(chemin):
    """Nos chunks du module 2 -> des Document LangChain."""
    documents = []
    for c in ingerer(chemin):
        metadata = {"source": c["source"], "section": c["section"], "methode": c["methode"]}
        if c["pages"]:  # Chroma refuse les listes vides
            metadata["pages"] = c["pages"]
        documents.append(Document(page_content=c["texte"], metadata=metadata, id=c["id"]))
    return documents


def indexer_si_vide():
    if base.get(limit=1)["ids"]:
        return
    documents = [d for f in sorted(DOSSIER_DATA.iterdir()) if f.suffix.lower() in (".pdf", ".txt") for d in vers_documents(f)]
    base.add_documents(documents, ids=[d.id for d in documents])
    print(f"Index construit : {len(documents)} chunks.")


# ---------- La chaîne RAG (LCEL) ----------
def decrire_source(metadata):
    pages = metadata.get("pages")
    if not pages:
        return metadata["source"]
    if len(pages) == 1:
        return f"{metadata['source']}, page {pages[0]}"
    return f"{metadata['source']}, pages {pages[0]}-{pages[-1]}"


def formater(documents):
    return "\n\n".join(
        f"[{n}] (source : {decrire_source(d.metadata)})\n{d.page_content}" for n, d in enumerate(documents, start=1)
    )


def creer_chaine(filtre=None):
    search_kwargs = {"k": TOP_K} if filtre is None else {"k": TOP_K, "filter": filtre}
    retriever = base.as_retriever(search_kwargs=search_kwargs)
    return (
        RunnableParallel(documents=retriever, question=RunnablePassthrough())
        .assign(contexte=lambda x: formater(x["documents"]))
        .assign(reponse=prompt | llm | StrOutputParser())
    )


def repondre(question, filtre=None):
    documents = []
    for morceau in creer_chaine(filtre).stream(question):
        if "documents" in morceau:
            documents = morceau["documents"]
        if "reponse" in morceau:
            print(morceau["reponse"], end="", flush=True)
    print("\n\nSources :")
    for n, d in enumerate(documents, start=1):
        print(f"  [{n}] {decrire_source(d.metadata)}  ({d.id})")


if __name__ == "__main__":
    indexer_si_vide()
    while True:
        saisie = input("\nQuestion (q pour quitter, @fichier pour filtrer) : ").strip()
        if saisie.lower() == "q":
            break
        if saisie.startswith("@") and " " in saisie:
            source, question = saisie[1:].split(" ", 1)
            repondre(question, filtre={"source": source})
        elif saisie:
            repondre(saisie)