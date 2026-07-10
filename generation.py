import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path, override=True)


_CITATION_PATTERN = re.compile(r"\[S(\d+)\]")

# Single capital letters (initials) and common abbreviations that should not
# trigger sentence splits when followed by ". " — e.g. "J. F. Kennedy", "Dr. Smith"
_ABBREV_PATTERN = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Gov|Sen|Rep|Capt|Lt|Col|Gen|Maj|Rev|Hon|[A-Z])\."
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text):
    protected = _ABBREV_PATTERN.sub(lambda m: m.group().replace(".", "\x00"), text)
    parts = _SENTENCE_SPLIT.split(protected)
    return [p.strip().replace("\x00", ".") for p in parts if p.strip()]


class AnswerGenerator:
    def __init__(self, config=None):
        config = config or {}
        self.model = config.get("model") or os.environ.get("LLM_MODEL") or "gemma-4-31B-it"
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 1024)
        self.api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        self.api_base = config.get("api_base") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        self.extra_headers = config.get("headers") or {}

    def _call_llm(self, messages):
        url = f"{self.api_base.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        resp = requests.post(url, headers=headers, json=body, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"].get("content")
        reasoning = data["choices"][0]["message"].get("reasoning_content")
        if content:
            return content
        if reasoning:
            return reasoning
        return ""

    def generate(self, query_text, context_blocks, chat_history=None):
        if chat_history is None:
            chat_history = []
            
        if not context_blocks:
            return {
                "answer_text": None,
                "citations": [],
                "grounded": True,
                "abstained": True,
                "reason": "No context blocks provided",
            }

        context_str = self._format_context(context_blocks)
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        
        # Append chat history
        for msg in chat_history[-5:]: # Keep last 5 messages for context size
            messages.append({"role": msg["role"], "content": msg["content"]})
            
        # Append the new query with context
        messages.append({"role": "user", "content": f"Context blocks:\n\n{context_str}\n\nQuestion: {query_text}"})

        raw = self._call_llm(messages)

        abstained = raw.strip().upper().startswith("ABSTAIN:")
        conflict = raw.strip().upper().startswith("CONFLICT:")

        if abstained:
            return {
                "answer_text": None,
                "citations": [],
                "grounded": True,
                "abstained": True,
                "reason": raw.strip()[len("ABSTAIN:"):].strip() or None,
            }

        if conflict:
            return {
                "answer_text": raw.strip(),
                "citations": [],
                "grounded": False,
                "abstained": False,
                "reason": raw.strip(),
            }

        citations = self._extract_citations(raw, context_blocks)
        grounded, reason = self._validate_citations(raw, citations, context_blocks)

        return {
            "answer_text": raw.strip(),
            "citations": citations,
            "grounded": grounded,
            "abstained": False,
            "reason": reason,
        }

    def _format_context(self, context_blocks):
        parts = []
        for i, block in enumerate(context_blocks):
            block_id = f"[S{i:02d}]"
            parts.append(f"{block_id} {block['text']}")
        return "\n\n".join(parts)

    def _extract_citations(self, answer_text, context_blocks):
        seen = set()
        citations = []
        for match in _CITATION_PATTERN.finditer(answer_text):
            idx = int(match.group(1))
            cid = f"S{idx:02d}"
            if cid in seen:
                continue
            seen.add(cid)
            clamped_idx = min(idx, len(context_blocks) - 1) if context_blocks else idx
            if clamped_idx < len(context_blocks):
                block = context_blocks[clamped_idx]
                citations.append({
                    "citation_id": cid,
                    "source_id": block.get("source_id"),
                    "section_id": block.get("section_id"),
                    "supporting_child_ids": block.get("supporting_child_ids", []),
                })
        return citations

    def _validate_citations(self, answer_text, citations, context_blocks):
        reasons = []

        cited_indices = {int(m.group(1)) for m in _CITATION_PATTERN.finditer(answer_text)}
        out_of_bounds = [i for i in cited_indices if i >= len(context_blocks)]
        if out_of_bounds:
            reasons.append(f"Citation indices out of bounds: {out_of_bounds}")

        sentences = _split_sentences(answer_text)
        factual_sentences = [
            s for s in sentences
            if not s.upper().startswith(("ABSTAIN:", "CONFLICT:"))
        ]
        uncited_count = sum(1 for s in factual_sentences if not _CITATION_PATTERN.search(s))
        if uncited_count > 0:
            reasons.append(f"Sentences without citations: {uncited_count}")

        if not citations and not any(
            answer_text.strip().upper().startswith(p) for p in ("ABSTAIN:", "CONFLICT:")
        ):
            reasons.append("No citations found in generated answer")

        grounded = len(reasons) == 0
        reason = "; ".join(reasons) if reasons else None
        return grounded, reason


def build_context_blocks(sections):
    blocks = []
    for section in sections:
        blocks.append(_make_context_block(section))
    return blocks


def build_context_blocks_from_children(reranked_children):
    blocks = []
    for child in reranked_children:
        parent_id = child.get("parent_id")
        block = {
            "source_id": child.get("doc_id"),
            "section_id": parent_id or child.get("chunk_id"),
            "section_path": child.get("section_path"),
            "text": child.get("text", ""),
            "supporting_child_ids": [child["chunk_id"]] if child.get("chunk_type") == "chunk" else [],
            "retrieval_score": child.get("retrieval_score", 0.0),
            "rerank_score": child.get("rerank_score", child.get("score", 0.0)),
            "chunk_type": child.get("chunk_type"),
            "title": child.get("title"),
            "source_url": child.get("source_url"),
            "parent_id": parent_id,
            "keywords": child.get("keywords", []),
            "topics": child.get("topics", []),
            "doshas": child.get("doshas", []),
            "symptoms": child.get("symptoms", []),
            "treatments": child.get("treatments", []),
        }
        blocks.append(block)
    return blocks


def _make_context_block(item):
    return {
        "source_id": item.get("doc_id"),
        "section_id": item.get("chunk_id"),
        "section_path": item.get("section_path"),
        "text": item.get("text", ""),
        "supporting_child_ids": item.get("child_ids", []),
        "retrieval_score": item.get("retrieval_score", 0.0),
        "rerank_score": item.get("rerank_score", item.get("score", 0.0)),
        "chunk_type": item.get("chunk_type"),
        "title": item.get("title"),
        "source_url": item.get("source_url"),
        "parent_id": item.get("parent_id"),
        "keywords": item.get("keywords", []),
        "topics": item.get("topics", []),
        "doshas": item.get("doshas", []),
        "symptoms": item.get("symptoms", []),
        "treatments": item.get("treatments", []),
    }


_SYSTEM_PROMPT = """\
You are an Ayurvedic medicine educational assistant. Follow these rules strictly:

1. Answer the question using ONLY facts from the provided context blocks.
2. Each context block is prefixed with its index like [S00], [S01], etc.
3. Every factual sentence in your answer MUST end with the citation marker of the \
context block(s) it uses. Example: "Turmeric has anti-inflammatory properties. [S00]"
4. If multiple blocks support the same claim, list all: [S00][S02]
5. If the context blocks do NOT contain sufficient evidence to answer, respond with \
exactly: ABSTAIN:
6. If the context blocks contain conflicting information, state the conflict clearly \
and respond with: CONFLICT:
7. Do not use any knowledge outside the provided context blocks.
8. Do not output a citation unless the cited block directly supports the claim.
9. Keep answers concise and factual.
10. When discussing treatments, include a disclaimer that this is educational \
information and not medical advice. Patients should consult a qualified Ayurvedic practitioner.
11. Reference Ayurvedic concepts (doshas, dhatus, agni, etc.) using their Sanskrit \
names with brief English explanations where helpful."""
