# Análise de avaliações de clientes

Classifica avaliações de clientes (positiva, neutra ou negativa, mais o tópico: entrega, preço, qualidade, atendimento ou app) e mostra um dashboard. Também compara um modelo clássico (TF-IDF + regressão logística) com um LLM.

## Como rodar

```bash
pip install -r requirements.txt
python -m review_insights.data_gen    # gera data/reviews.csv e data/hard.csv
python compare.py                     # baseline sempre; LLM se tiver ANTHROPIC_API_KEY
streamlit run app.py
python -m pytest tests
```

## Resultado do modelo clássico

| Conjunto | Acurácia | F1 macro |
|---|---|---|
| Teste (dados gerados por mim) | 0,967 | 0,962 |
| Difícil (15 frases que escrevi, com ironia e sentimento misto) | 0,333 | 0,336 |

O modelo vai muito bem em frases parecidas com as de treino e erra quase tudo nas com ironia ("Adorei esperar 20 dias para receber"). O `compare.py` já está pronto para colocar um LLM lado a lado com esse modelo, medindo acurácia, tempo e custo.

## Limitações

- O dataset principal é sintético (montei com frases-modelo e 5% de rótulos trocados), por isso a acurácia é alta demais.
- O conjunto difícil tem só 15 frases.

## Próximos passos

- Trocar o dataset por avaliações reais
- Rodar o `compare.py` com um LLM e registrar a tabela aqui
