import os
from collections import defaultdict

import numpy as np
import torch
from sentence_transformers import CrossEncoder

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-2-v2"


def _get_device():
    if not torch.cuda.is_available():
        return "cpu"
    major, _ = torch.cuda.get_device_capability()
    if major >= 7:
        return "cuda"
    return "cpu"


def _get_onnx_path(model_name):
    safe = model_name.replace("/", "__").replace("\\", "__")
    cache_dir = os.environ.get("ONNX_CACHE_DIR", "data")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{safe}.onnx")


def _export_cross_encoder(ce, onnx_path, device):
    tokenizer = ce.tokenizer
    config = ce.config
    max_len = getattr(config, "max_length", 512)

    texts = [
        "sample query text for ONNX export tracing with more realistic length",
        "This is a longer sample document passage that better represents actual content "
        "that would be processed during reranking. It includes multiple sentences to "
        "ensure the traced model handles variable sequence lengths correctly.",
    ]
    features = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    features = {k: v.to(device) for k, v in features.items()}

    has_token_type = "token_type_ids" in features

    hf_model = ce.model
    hf_model.eval()

    if has_token_type:
        input_tensors = (features["input_ids"], features["attention_mask"], features["token_type_ids"])
        input_names = ["input_ids", "attention_mask", "token_type_ids"]
    else:
        input_tensors = (features["input_ids"], features["attention_mask"])
        input_names = ["input_ids", "attention_mask"]

    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len"},
        "logits": {0: "batch"},
    }
    if has_token_type:
        dynamic_axes["token_type_ids"] = {0: "batch", 1: "seq_len"}

    torch.onnx.export(
        hf_model,
        input_tensors,
        onnx_path,
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
    )

    hf_model.train()


class Reranker:
    def __init__(self, model_name=None):
        if model_name is None:
            model_name = os.environ.get("RERANK_MODEL", DEFAULT_MODEL)
        device = _get_device()
        self.device = device
        self.model_name = model_name

        ce = CrossEncoder(model_name, device=device)
        self.tokenizer = ce.tokenizer
        self._onnx_session = None
        self.model = ce
        self._backend = "pytorch"

        if device == "cpu":
            self._init_onnx(ce)

        backend_tag = f" ({self._backend})" if device == "cpu" else ""
        print(f"  Reranker loaded on {device}{backend_tag}")

    def _init_onnx(self, ce):
        try:
            import onnxruntime as ort
        except ImportError:
            print("  [reranker] onnxruntime not installed, using pytorch")
            return

        onnx_path = _get_onnx_path(self.model_name)

        if os.path.exists(onnx_path) and os.path.getsize(onnx_path) < 1024:
            os.remove(onnx_path)

        if not os.path.exists(onnx_path):
            print(f"  [reranker] exporting ONNX model to {onnx_path} ...")
            try:
                _export_cross_encoder(ce, onnx_path, self.device)
            except Exception as e:
                print(f"  [reranker] ONNX export failed ({e}), using pytorch")
                return

        try:
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_opts.intra_op_num_threads = max(os.cpu_count() or 4, 1)
            sess_opts.inter_op_num_threads = 1
            self._onnx_session = ort.InferenceSession(
                onnx_path, sess_opts, providers=["CPUExecutionProvider"],
            )
            self._backend = "onnx"
        except Exception as e:
            self._onnx_session = None
            print(f"  [reranker] ONNX session failed ({e}), using pytorch")

    def _tokenize(self, texts, docs):
        config = self.model.config
        max_len = getattr(config, "max_length", 512)
        return self.tokenizer(
            texts, docs,
            padding=True, truncation=True, max_length=max_len, return_tensors="np",
        )

    def _predict_pytorch(self, pairs):
        return self.model.predict(pairs, show_progress_bar=False, batch_size=128)

    def _predict_onnx(self, pairs):
        import sys
        texts = [p[0] for p in pairs]
        docs = [p[1] for p in pairs]

        batch_size = 64
        all_scores = []

        print(f"  [reranker] ONNX inference starting: {len(pairs)} pairs in { (len(pairs) + batch_size - 1) // batch_size } batches", flush=True)

        for i in range(0, len(pairs), batch_size):
            print(f"  [reranker]   batch {i // batch_size + 1}...", end="", flush=True)
            batch_texts = texts[i:i + batch_size]
            batch_docs = docs[i:i + batch_size]
            tokenized = self._tokenize(batch_texts, batch_docs)

            ort_inputs = {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
            }
            if "token_type_ids" in tokenized:
                ort_inputs["token_type_ids"] = tokenized["token_type_ids"]

            logits = self._onnx_session.run(["logits"], ort_inputs)[0]
            all_scores.append(logits)
            print(" ok", flush=True)

        print(f"  [reranker] concatenating {len(all_scores)} score arrays...", end="", flush=True)
        logits = np.concatenate(all_scores, axis=0)
        print(" ok", flush=True)
        result = torch.sigmoid(torch.from_numpy(logits)).numpy().flatten()
        print(f"  [reranker] ONNX inference complete: {len(result)} scores", flush=True)
        return result

    def _predict(self, pairs):
        if self._onnx_session is not None:
            print(f"  [reranker] using ONNX backend for {len(pairs)} pairs", flush=True)
            try:
                return self._predict_onnx(pairs)
            except Exception as e:
                print(f"  [reranker] ONNX inference failed ({e}), falling back to pytorch", flush=True)
                self._onnx_session = None
                self._backend = "pytorch"
        return self._predict_pytorch(pairs)

    def rerank(self, query, candidates, top_k=8):
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self._predict(pairs)

        for c, score in zip(candidates, scores):
            if "retrieval_score" not in c:
                c["retrieval_score"] = c.get("score", 0.0)
            c["rerank_score"] = float(score)

        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates[:top_k]

    def rerank_batch(self, query_candidates_pairs, top_k=8):
        import sys
        if not query_candidates_pairs:
            return []

        print(f"  [reranker] rerank_batch: building pairs from {len(query_candidates_pairs)} queries...", end="", flush=True)
        all_pairs = []
        flat_candidates = []
        for query, candidates in query_candidates_pairs:
            flat_candidates.append(candidates)
            all_pairs.extend([(query, c["text"]) for c in candidates])
        print(f" {len(all_pairs)} pairs", flush=True)

        if not all_pairs:
            return [[] for _ in query_candidates_pairs]

        all_scores = self._predict(all_pairs)

        print(f"  [reranker] sorting and collecting top-{top_k} per query...", end="", flush=True)

        results = []
        idx = 0
        for candidates in flat_candidates:
            count = len(candidates)
            scores = all_scores[idx:idx + count]

            for c, score in zip(candidates, scores):
                if "retrieval_score" not in c:
                    c["retrieval_score"] = c.get("score", 0.0)
                c["rerank_score"] = float(score)

            scored = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
            results.append(scored[:top_k])
            idx += count

        print(" ok", flush=True)
        return results


def assemble_neighbor_context(child_results, db, window_size=2, top_k=None):
    if not child_results:
        return []

    results = []
    seen_ids = set()
    children = child_results[:top_k] if top_k else child_results

    for child in children:
        chunk_id = child["chunk_id"]
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)

        prev_chunks = _walk_prev(child, db, window_size)
        next_chunks = _walk_next(child, db, window_size)
        all_chunks = prev_chunks + [child] + next_chunks
        assembled_text = "\n\n".join(c["text"] for c in all_chunks)

        results.append({
            "chunk_id": child["chunk_id"],
            "text": assembled_text,
            "score": child.get("rerank_score", child.get("score", 0.0)),
            "rerank_score": child.get("rerank_score", child.get("score", 0.0)),
            "retrieval_score": child.get("retrieval_score", child.get("score", 0.0)),
            "source_id": child.get("doc_id"),
            "section_id": child.get("chunk_id"),
            "child_ids": [child["chunk_id"]],
            "supporting_child_ids": [child["chunk_id"]],
            "doc_id": child.get("doc_id"),
            "chunk_type": child.get("chunk_type"),
            "title": child.get("title"),
            "source_url": child.get("source_url"),
            "parent_id": child.get("parent_id"),
            "section_path": child.get("section_path"),
            "keywords": child.get("keywords", []),
            "topics": child.get("topics", []),
            "doshas": child.get("doshas", []),
            "symptoms": child.get("symptoms", []),
            "treatments": child.get("treatments", []),
        })

    return results


def _walk_prev(chunk, db, steps):
    prevs = []
    prev_id = chunk.get("prev_id")
    for _ in range(steps):
        if not prev_id:
            break
        prev = db.get_chunk(prev_id)
        if prev is None:
            break
        prevs.insert(0, prev)
        prev_id = prev.get("prev_id")
    return prevs


def _walk_next(chunk, db, steps):
    nexts = []
    next_id = chunk.get("next_id")
    for _ in range(steps):
        if not next_id:
            break
        nxt = db.get_chunk(next_id)
        if nxt is None:
            break
        nexts.append(nxt)
        next_id = nxt.get("next_id")
    return nexts
