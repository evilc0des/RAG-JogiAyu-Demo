from collections import defaultdict

import torch
from sentence_transformers import CrossEncoder


def _get_device():
    if not torch.cuda.is_available():
        return "cpu"
    major, _ = torch.cuda.get_device_capability()
    if major >= 7:
        return "cuda"
    return "cpu"


class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        device = _get_device()
        self.model = CrossEncoder(model_name, device=device)
        print(f"  Reranker loaded on {device}")

    def rerank(self, query, candidates, top_k=8):
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
            batch_size=min(len(pairs), 64),
        )

        for c, score in zip(candidates, scores):
            if "retrieval_score" not in c:
                c["retrieval_score"] = c.get("score", 0.0)
            c["rerank_score"] = float(score)

        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates[:top_k]

    def rerank_batch(self, query_candidates_pairs, top_k=8):
        if not query_candidates_pairs:
            return []

        all_pairs = []
        counts = []
        flat_candidates = []
        for query, candidates in query_candidates_pairs:
            counts.append(len(candidates))
            flat_candidates.append(candidates)
            all_pairs.extend([(query, c["text"]) for c in candidates])

        if not all_pairs:
            return [[] for _ in query_candidates_pairs]

        all_scores = self.model.predict(
            all_pairs,
            show_progress_bar=False,
            batch_size=128,
        )

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
