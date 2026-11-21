import json
import sys
import time
from pathlib import Path
import fasttext

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
model_path = Path(config["model_path"])
meta_path = model_path.with_suffix(".json")
metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
if (model_path.exists() and metadata.get("signature") == config["signature"]
        and not config["force_retrain"]):
    model = fasttext.load_model(str(model_path))
    print("Using cached supervised FastText model.", flush=True)
else:
    print("Training supervised FastText with Wiki initialization...", flush=True)
    start = time.perf_counter()
    model = fasttext.train_supervised(
        input=config["train_path"], lr=config["lr"], epoch=config["epochs"],
        dim=config["dimension"], wordNgrams=2, minn=2, maxn=5,
        bucket=config["bucket"], minCount=1, loss="softmax", thread=1,
        pretrainedVectors=config["pretrained_path"], verbose=0,
    )
    model.save_model(str(model_path))
    metadata = {"signature": config["signature"],
                "training_seconds": time.perf_counter() - start,
                "best_epoch": config["epochs"], "thread": 1,
                "wiki_initialization": True}
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

def top_labels(path):
    texts = Path(path).read_text(encoding="utf-8").splitlines()
    predictions = []
    for start in range(0, len(texts), 256):
        batch = texts[start:start + 256]
        try:
            labels, _ = model.predict(batch, k=1)
        except ValueError as error:
            # FastText 0.9.2's public wrapper uses np.array(copy=False), which
            # NumPy 2 rejects. The underlying FastText predictions are identical.
            if "copy" not in str(error).lower():
                raise
            labels, _ = model.f.multilinePredict([text + "\n" for text in batch], 1, 0.0, "strict")
        predictions.extend(row[0].removeprefix("__label__") for row in labels)
    return predictions

output = {"validation": top_labels(config["validation_path"]),
          "test": top_labels(config["test_path"]), "metadata": metadata}
Path(config["predictions_path"]).write_text(json.dumps(output), encoding="utf-8")
print("FastText predictions saved.", flush=True)
