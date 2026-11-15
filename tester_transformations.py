from evaluer_recherche import JEU_DE_TEST, rang
from recherche import recherche_dense
from transformer_question import recherche_hyde, recherche_multi_query

for question, bons in [JEU_DE_TEST[i] for i in (3, 6, 15)]:
    docs_mq, variantes = recherche_multi_query(question)
    docs_hyde, imagine = recherche_hyde(question)
    print(f"\n{question}")
    print(f"  rang : dense={rang(recherche_dense(question), bons)}  multi-query={rang(docs_mq, bons)}  HyDE={rang(docs_hyde, bons)}")
    print(f"  variantes : {variantes[1:]}")
    print(f"  HyDE : {imagine[:150]!r}")