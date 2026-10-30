import random
from pathlib import Path

import pandas as pd

TOPICS = {
    "entrega": {
        "positive": ["Chegou muito antes do prazo", "Entrega rápida e embalagem perfeita", "Recebi no dia seguinte, adorei"],
        "neutral": ["A entrega ocorreu dentro do prazo informado", "Chegou na data prevista", "Entrega normal, sem novidades"],
        "negative": ["Atrasou duas semanas e ninguém avisou", "Veio com a caixa toda amassada", "Entrega absurdamente demorada"],
    },
    "preço": {
        "positive": ["Preço excelente pelo que entrega", "Muito barato para a qualidade", "Valeu cada centavo, ótimo custo-benefício"],
        "neutral": ["O preço está na média do mercado", "Valor razoável, nada demais", "Preço igual ao das outras lojas"],
        "negative": ["Caro demais para o que oferece", "Preço abusivo, não compensa", "Paguei muito e esperava mais"],
    },
    "qualidade": {
        "positive": ["Produto de ótima qualidade, superou expectativas", "Acabamento impecável", "Muito bem feito e resistente"],
        "neutral": ["O produto é como na descrição", "Qualidade mediana, cumpre o básico", "Produto comum, funciona"],
        "negative": ["Quebrou na primeira semana", "Material fraco, parece de brinquedo", "Qualidade péssima, veio com defeito"],
    },
    "atendimento": {
        "positive": ["Atendimento nota dez, resolveram na hora", "Equipe educada e prestativa", "O suporte foi super atencioso"],
        "neutral": ["Fui atendido, sem problemas nem elogios", "O suporte respondeu em um dia útil", "Atendimento padrão"],
        "negative": ["Atendimento horrível, ninguém resolve nada", "Fui ignorado pelo suporte por dias", "Atendentes grosseiros e despreparados"],
    },
    "app": {
        "positive": ["O app é rápido e intuitivo", "Aplicativo muito fácil de usar", "Adorei a nova versão do app"],
        "neutral": ["O app funciona, interface comum", "Aplicativo simples, faz o que promete", "App sem novidades nesta versão"],
        "negative": ["O app trava toda hora", "Aplicativo lento e cheio de bugs", "Não consigo logar no app de jeito nenhum"],
    },
}
SUFFIX = ["", "", " Recomendo.", " No geral é isso.", " Pretendo comprar de novo.", " Fica o registro.", " Foi minha experiência."]
PREFIX = ["", "", "Olha, ", "Sinceramente, ", "Resumindo: "]

# Casos que enganam modelos baseados em palavras (ironia, negação, sentimento misto).
HARD = [
    ("Ótimo, mais uma vez o produto chegou quebrado. Parabéns pela eficiência.", "negative", "qualidade"),
    ("Adorei esperar 20 dias para receber. Muito rápido mesmo.", "negative", "entrega"),
    ("Não é barato, mas também não posso dizer que seja caro.", "neutral", "preço"),
    ("O app não é ruim, só poderia ser mais rápido.", "neutral", "app"),
    ("Entrega rápida, só que o produto veio com defeito. Fiquei na dúvida do que sentir.", "negative", "qualidade"),
    ("Esperava pior, o atendimento acabou me surpreendendo positivamente.", "positive", "atendimento"),
    ("Nunca tive tantos problemas para ser tão bem atendido, equipe incrível.", "positive", "atendimento"),
    ("Pelo preço, não dá para reclamar. Cumpre o que promete.", "positive", "preço"),
    ("Se você gosta de esperar, essa loja é perfeita.", "negative", "entrega"),
    ("O produto é igual ao da foto. Nada a acrescentar.", "neutral", "qualidade"),
    ("Funciona. Só isso.", "neutral", "qualidade"),
    ("Cobrança duplicada e ainda me pedem paciência. Sensacional.", "negative", "atendimento"),
    ("Achei o app bom, mas o login falhou três vezes hoje.", "negative", "app"),
    ("Paguei menos que nas outras lojas e veio perfeito.", "positive", "preço"),
    ("Chegou no prazo exato que prometeram.", "neutral", "entrega"),
]

def make_dataset(n: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        topic = rng.choice(list(TOPICS))
        sentiment = rng.choices(["positive", "neutral", "negative"], weights=[0.4, 0.2, 0.4])[0]
        text = rng.choice(PREFIX) + rng.choice(TOPICS[topic][sentiment]) + "." + rng.choice(SUFFIX)
        if rng.random() < 0.05:  # ruído de rotulagem, como em dados reais
            sentiment = rng.choice(["positive", "neutral", "negative"])
        month = 1 + (i * 12) // n
        rows.append({"review": text.strip(), "sentiment": sentiment, "topic": topic, "date": f"2026-{month:02d}-15"})
    return pd.DataFrame(rows)

def make_hard() -> pd.DataFrame:
    return pd.DataFrame(HARD, columns=["review", "sentiment", "topic"])

if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data"
    out.mkdir(exist_ok=True)
    make_dataset().to_csv(out / "reviews.csv", index=False)
    make_hard().to_csv(out / "hard.csv", index=False)
    print("dados gerados em", out)
