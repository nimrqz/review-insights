import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.model_selection import train_test_split  # noqa: E402

from review_insights import llm_classifier  # noqa: E402
from review_insights.baseline import train  # noqa: E402
from review_insights.data_gen import make_dataset, make_hard  # noqa: E402
from review_insights.llm_classifier import Batch, Item, classify  # noqa: E402

class FakeClient:
    def __init__(self, parsed_by_call):
        self.parsed_by_call, self.calls = list(parsed_by_call), []
        self.messages = NS(parse=self._parse)

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        return NS(parsed_output=self.parsed_by_call.pop(0), usage=NS(input_tokens=100, output_tokens=20))

def test_dataset_is_reproducible_and_balanced_enough():
    a, b = make_dataset(100, seed=1), make_dataset(100, seed=1)
    assert a.equals(b)
    assert set(a.sentiment) == {"positive", "neutral", "negative"}

def test_baseline_learns_synthetic_but_is_not_perfect_on_hard_set():
    df = make_dataset(400)
    tr, te = train_test_split(df, test_size=0.25, random_state=0, stratify=df.sentiment)
    model = train(tr.review, tr.sentiment)
    assert (model.predict(te.review) == te.sentiment).mean() > 0.8
    hard = make_hard()
    hard_acc = (model.predict(hard.review) == hard.sentiment).mean()
    assert hard_acc < 0.9  # o conjunto difícil existe justamente para expor o baseline

def test_llm_classifier_maps_ids_and_batches(monkeypatch):
    monkeypatch.setattr(llm_classifier, "BATCH_SIZE", 2)
    client = FakeClient([
        Batch(items=[Item(id=0, sentiment="negative", topic="entrega"), Item(id=1, sentiment="positive", topic="app")]),
        Batch(items=[Item(id=0, sentiment="neutral", topic="preço"), Item(id=9, sentiment="positive", topic="app")]),
    ])
    result = classify(["a", "b", "c"], client=client)
    assert result.sentiments == ["negative", "positive", "neutral"]
    assert result.topics == ["entrega", "app", "preço"]
    assert len(client.calls) == 2 and result.input_tokens == 200 and result.output_tokens == 40

def test_llm_classifier_survives_refusal():
    result = classify(["a", "b"], client=FakeClient([None]))
    assert result.sentiments == ["neutral", "neutral"]
