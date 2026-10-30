from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

LABELS = ["negative", "neutral", "positive"]

def train(texts, labels) -> Pipeline:
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode", min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    return model.fit(texts, labels)
