from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

llm = ChatOllama(model="qwen2.5:7b", temperature=0, num_ctx=8192)

# 1. Appeler le modèle directement
message = llm.invoke("Dis bonjour en darija, en une phrase.")
print("1.", type(message).__name__, "->", message.content)

# 2. Un prompt avec des variables
prompt = ChatPromptTemplate.from_messages([
    ("system", "Tu es un professeur. Tu expliques en {nb_phrases} phrases simples, en français."),
    ("human", "Explique : {notion}"),
])
print("2.", prompt.invoke({"nb_phrases": 2, "notion": "un embedding"}).to_messages())

# 3. Une chaîne LCEL : prompt | modèle | parseur
chaine = prompt | llm | StrOutputParser()
print("3.", chaine.invoke({"nb_phrases": 2, "notion": "un embedding"}))

# 4. Le streaming : la même chaîne, morceau par morceau
print("4. ", end="")
for morceau in chaine.stream({"nb_phrases": 1, "notion": "le chunking"}):
    print(morceau, end="", flush=True)
print()

# 5. Le batch : plusieurs entrées d'un coup
for r in chaine.batch([{"nb_phrases": 1, "notion": n} for n in ["HNSW", "BM25"]]):
    print("5.", r)


# 6. Une sortie structurée : le modèle remplit un objet Python
class Fiche(BaseModel):
    notion: str = Field(description="Le nom de la notion")
    definition: str = Field(description="Une définition en une phrase")
    exemple: str = Field(description="Un exemple concret")
    difficulte: int = Field(description="Difficulté de 1 (facile) à 5 (difficile)")


extracteur = ChatPromptTemplate.from_messages([("human", "Fais une fiche de révision sur : {notion}")]) | llm.with_structured_output(Fiche)
fiche = extracteur.invoke({"notion": "la similarité cosinus"})
print("6.", type(fiche).__name__, fiche)