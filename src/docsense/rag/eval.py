"""RAG evaluation harness: retrieval metrics + LLM-judged answer quality.

Each configuration (chunk size, k, hybrid on/off) is one MLflow run, so the
chunking/retrieval design space is explored with the same rigor as model
hyperparameters.

Gold data: data/eval/qa_pairs.jsonl, one object per line:
    {"question": ..., "answer": ..., "doc_id": ..., "page": ...}

Usage:
    python -m docsense.rag.eval [--judge] [--top-k 5] [--no-hybrid]
"""

from __future__ import annotations

import argparse
import json
import logging

import mlflow

from docsense.llm.factory import get_provider
from docsense.rag.chain import ask
from docsense.retrieval.hybrid import retrieve
from docsense.settings import get_config, get_settings, resolve_path

logger = logging.getLogger(__name__)


def load_qa_pairs() -> list[dict]:
    path = resolve_path(get_config()["eval"]["qa_pairs_path"])
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def retrieval_metrics(pairs: list[dict], top_k: int) -> dict[str, float]:
    """Hit-rate@k and MRR@k: does a chunk from the gold (doc, page) surface?"""
    hits_at_k, reciprocal_ranks = 0, []
    for pair in pairs:
        results = retrieve(pair["question"], top_k=top_k)
        rank = next(
            (
                i + 1
                for i, hit in enumerate(results)
                if hit.chunk.doc_id == pair["doc_id"]
                and (pair.get("page") is None or hit.chunk.page == pair["page"])
            ),
            None,
        )
        if rank is not None:
            hits_at_k += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    n = max(len(pairs), 1)
    return {
        f"hit_rate_at_{top_k}": round(hits_at_k / n, 4),
        f"mrr_at_{top_k}": round(sum(reciprocal_ranks) / n, 4),
    }


def judge_answers(pairs: list[dict], judge_provider=None) -> dict[str, float]:
    """Answer each gold question via the RAG chain, grade with an LLM judge."""
    judge_provider = judge_provider or get_provider()
    template = resolve_path(get_config()["rag"]["judge_prompt_path"]).read_text(encoding="utf-8")
    correct = 0
    for pair in pairs:
        result = ask(pair["question"])
        grade = judge_provider.complete(
            template.format(
                question=pair["question"], reference=pair["answer"], candidate=result.answer
            ),
            max_tokens=8,
        )
        if "CORRECT" in grade.upper() and "INCORRECT" not in grade.upper():
            correct += 1
    return {"judged_accuracy": round(correct / max(len(pairs), 1), 4)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument("--judge", action="store_true", help="also run LLM-judged answer accuracy")
    args = parser.parse_args()

    cfg = get_config()
    top_k = args.top_k or cfg["retrieval"]["top_k"]
    if args.no_hybrid:
        cfg["retrieval"]["hybrid"] = False

    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(cfg["eval"]["experiment_name"])

    pairs = load_qa_pairs()
    with mlflow.start_run():
        mlflow.log_params(
            {
                "top_k": top_k,
                "hybrid": cfg["retrieval"]["hybrid"],
                "chunk_size": cfg["indexing"]["chunk_size"],
                "chunk_overlap": cfg["indexing"]["chunk_overlap"],
                "embedding_model": cfg["indexing"]["embedding_model"],
                "n_qa_pairs": len(pairs),
            }
        )
        metrics = retrieval_metrics(pairs, top_k)
        if args.judge:
            metrics |= judge_answers(pairs)
        mlflow.log_metrics(metrics)
        logger.info("Eval metrics: %s", metrics)


if __name__ == "__main__":
    main()
