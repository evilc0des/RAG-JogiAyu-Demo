import json
import re


class MultiHopOrchestrator:
    def __init__(self, config=None):
        config = config or {}
        self.model = config.get("model", "gemma-4-31B-it")
        self.temperature = config.get("temperature", 0.0)
        self.max_hops = config.get("max_hops", 3)
        self.max_sub_queries_per_hop = config.get("max_sub_queries_per_hop", 3)
        self._generator = None

    def _get_generator(self):
        if self._generator is None:
            from generation import AnswerGenerator
            self._generator = AnswerGenerator({
                "model": self.model,
                "temperature": self.temperature,
            })
        return self._generator

    def reason(self, original_query, accumulated_context_text, hop_number, previous_sub_queries, chat_history=None):
        generator = self._get_generator()
        reasoning_messages = [{"role": "system", "content": _REASONING_SYSTEM_PROMPT}]
        if chat_history:
            for msg in chat_history[-5:]:
                reasoning_messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        reasoning_messages.append({"role": "user", "content": _build_reasoning_user_message(
            original_query, accumulated_context_text, hop_number, previous_sub_queries
        )})
        raw = generator._call_llm(reasoning_messages)
        return self._parse_action_json(raw, hop_number)

    def execute_search(self, sub_queries, sparse_retriever, dense_retriever, db):
        from retrieval import hybrid_retrieve_with_rerank

        all_sections = {}
        all_context_blocks = []
        seen_ids = set()

        for sq in sub_queries:
            result = hybrid_retrieve_with_rerank(
                sq, sparse_retriever, dense_retriever, db,
                rerank_top_k=5,
            )
            for section in result["results"]:
                chunk_id = section.get("chunk_id")
                if not chunk_id:
                    continue
                if chunk_id not in all_sections:
                    all_sections[chunk_id] = {
                        **section,
                        "source_queries": [sq],
                    }
                else:
                    existing = all_sections[chunk_id]
                    existing["source_queries"].append(sq)
                    if section.get("rerank_score", section.get("score", 0)) > existing.get("rerank_score", existing.get("score", 0)):
                        existing["rerank_score"] = section.get("rerank_score", section.get("score", 0))
                        existing["score"] = section.get("score", 0)
                        existing["child_ids"] = section.get("child_ids", [])
                    existing["child_ids"] = list(set(existing.get("child_ids", []) + section.get("child_ids", [])))

            for section in result["results"]:
                chunk_id = section.get("chunk_id")
                if chunk_id and chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    all_context_blocks.append(section)

        sections = list(all_sections.values())
        sections.sort(key=lambda s: s.get("rerank_score", s.get("score", 0)), reverse=True)

        section_ids = [s["chunk_id"] for s in sections]
        return sections, all_context_blocks, section_ids

    def run(self, original_query, sparse_retriever, dense_retriever, db, chat_history=None):
        hop_trace = []
        accumulated_context_blocks = []
        accumulated_sections = {}
        seen_section_ids = set()
        seen_context_ids = set()
        previous_sub_queries = []

        for hop_number in range(self.max_hops):
            if not accumulated_context_blocks:
                accumulated_context_text = "(no evidence yet)"
            else:
                accumulated_context_text = self._format_accumulated_context(accumulated_context_blocks)

            action = self.reason(
                original_query,
                accumulated_context_text,
                hop_number + 1,
                previous_sub_queries,
                chat_history=chat_history,
            )

            if action.get("action") == "answer":
                hop_trace.append({
                    "hop_number": hop_number + 1,
                    "sub_queries": [],
                    "retrieved_section_ids": [],
                    "action": "answer",
                })
                break

            sub_queries = action.get("sub_queries", [])
            if not sub_queries:
                hop_trace.append({
                    "hop_number": hop_number + 1,
                    "sub_queries": [],
                    "retrieved_section_ids": [],
                    "action": "answer",
                })
                break

            sub_queries = [sq for sq in sub_queries if sq not in previous_sub_queries]
            if not sub_queries:
                hop_trace.append({
                    "hop_number": hop_number + 1,
                    "sub_queries": [],
                    "retrieved_section_ids": [],
                    "action": "answer (all sub-queries were duplicates)",
                })
                break

            previous_sub_queries.extend(sub_queries)

            sections, new_context_blocks, section_ids = self.execute_search(
                sub_queries, sparse_retriever, dense_retriever, db,
            )

            new_ids = [sid for sid in section_ids if sid not in seen_section_ids]
            seen_section_ids.update(section_ids)

            hop_trace.append({
                "hop_number": hop_number + 1,
                "sub_queries": sub_queries,
                "retrieved_section_ids": new_ids,
                "action": "search",
            })

            for section in sections:
                sid = section.get("chunk_id")
                if sid and sid not in accumulated_sections:
                    accumulated_sections[sid] = section

            for block in new_context_blocks:
                bid = block.get("section_id")
                if bid and bid not in seen_context_ids:
                    seen_context_ids.add(bid)
                    accumulated_context_blocks.append(block)

        if not hop_trace or hop_trace[-1].get("action") != "answer":
            hop_trace.append({
                "hop_number": len(hop_trace) + 1,
                "sub_queries": [],
                "retrieved_section_ids": [],
                "action": "answer (max hops reached)",
            })

        all_sections = list(accumulated_sections.values())
        all_sections.sort(key=lambda s: s.get("rerank_score", s.get("score", 0)), reverse=True)

        return hop_trace, accumulated_context_blocks, all_sections

    def _parse_action_json(self, raw_text, hop_number):
        json_text = raw_text.strip()

        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", json_text, re.DOTALL)
        if fence_match:
            json_text = fence_match.group(1).strip()

        brace_match = re.search(r"\{.*\}", json_text, re.DOTALL)
        if brace_match:
            json_text = brace_match.group(0)

        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            return {"action": "answer"}

        if not isinstance(parsed, dict):
            return {"action": "answer"}

        action = parsed.get("action", "").lower()
        if action == "search":
            sub_queries = parsed.get("sub_queries", [])
            if not isinstance(sub_queries, list):
                return {"action": "answer"}
            sub_queries = [str(sq).strip() for sq in sub_queries if sq and str(sq).strip()]
            sub_queries = sub_queries[:self.max_sub_queries_per_hop]
            if not sub_queries:
                return {"action": "answer"}
            return {"action": "search", "sub_queries": sub_queries}
        else:
            return {"action": "answer"}

    def _format_accumulated_context(self, context_blocks):
        parts = []
        for block in context_blocks:
            text = block.get("text", "")
            source = block.get("source_id", "unknown")
            parts.append(f"[Source: {source}] {text}")
        return "\n\n---\n\n".join(parts)


_REASONING_SYSTEM_PROMPT = """\
You are a multi-hop reasoning agent. Your task is to gather evidence to answer a complex question by searching through a knowledge base.

Given:
- The ORIGINAL question the user asked
- Any EVIDENCE already retrieved from previous searches (may be empty on first hop)
- The current HOP NUMBER
- A list of sub-queries you have ALREADY asked in previous hops

Decide your next action. You must output EXACTLY one JSON object:

Option A — You need more evidence:
{"action": "search", "sub_queries": ["specific search query 1", "specific search query 2"]}

Option B — You have enough evidence to answer:
{"action": "answer"}

Rules:
1. Each sub-query must be self-contained (resolve all pronouns and references — use full proper names).
2. Do NOT repeat sub-queries from the "already asked" list.
3. Generate 1-3 sub-queries maximum per hop.
4. Be specific — use proper names, not vague terms like "the event" or "that person".
5. If the available evidence already answers the question, output {"action": "answer"}.
6. Output ONLY valid JSON. No markdown, no explanation, no surrounding text."""


def _build_reasoning_user_message(original_query, accumulated_context_text, hop_number, previous_sub_queries):
    prev_list = "\n".join(f"  - {sq}" for sq in previous_sub_queries) if previous_sub_queries else "  (none)"
    return f"""ORIGINAL QUESTION: {original_query}

CURRENT HOP: {hop_number}

PREVIOUSLY ASKED SUB-QUERIES:
{prev_list}

EVIDENCE RETRIEVED SO FAR:
{accumulated_context_text}"""
