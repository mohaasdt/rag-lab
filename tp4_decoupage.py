from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion import ingerer

PDF = "data/guide_stage_etranger.pdf"

# LangChain "par défaut" : un Document par page, puis découpage récursif
pages = PyMuPDFLoader(PDF).load()
decoupeur = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
chunks_langchain = decoupeur.split_documents(pages)

# Ton pipeline du module 2
mes_chunks = ingerer(Path(PDF))

print(f"{len(pages)} pages -> {len(chunks_langchain)} chunks LangChain | {len(mes_chunks)} chunks ingestion.py\n")
for c in chunks_langchain[:4]:
    print(f"--- page {c.metadata['page']} : {c.page_content[:90]!r} ... {c.page_content[-50:]!r}")
print()
for c in mes_chunks[:4]:
    print(f"--- pages {c['pages']} | {c['section']!r} : {c['texte'][:90]!r}")