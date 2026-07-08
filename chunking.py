import re
import uuid
import math
from dataclasses import dataclass, field
from pathlib import Path

PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


@dataclass
class ChunkConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    split_on_paragraphs: bool = True
    use_semantic: bool = True
    semantic_threshold: float = 0.5
    llm_metadata: bool = False


def create_chunk(chunk_type: str, text: str, **metadata) -> dict:
    return {
        "chunk_id": str(uuid.uuid4()),
        "doc_id": metadata.get("doc_id"),
        "chunk_type": chunk_type,
        "text": text,
        "section_path": metadata.get("section_path"),
        "title": metadata.get("title"),
        "source_url": metadata.get("source_url"),
        "paragraph_start": metadata.get("paragraph_start"),
        "paragraph_end": metadata.get("paragraph_end"),
        "prev_id": metadata.get("prev_id"),
        "next_id": metadata.get("next_id"),
        "parent_id": metadata.get("parent_id"),
        "children_ids": [],
        "keywords": metadata.get("keywords", []),
        "topics": metadata.get("topics", []),
        "doshas": metadata.get("doshas", []),
        "symptoms": metadata.get("symptoms", []),
        "treatments": metadata.get("treatments", []),
        "llm_metadata": metadata.get("llm_metadata", {}),
    }


class HybridChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64,
                 split_on_paragraphs: bool = True, use_semantic: bool = True,
                 semantic_threshold: float = 0.5, llm_metadata: bool = False):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.split_on_paragraphs = split_on_paragraphs
        self.use_semantic = use_semantic
        self.semantic_threshold = semantic_threshold
        self.llm_metadata = llm_metadata
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from fastembed import TextEmbedding
            self._embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        return self._embedder

    def _tokenize_text(self, text: str) -> list[str]:
        return re.findall(r"\S+", text)

    def _cosine_similarity(self, a, b) -> float:
        import numpy as np
        a_arr = np.asarray(a, dtype=float)
        b_arr = np.asarray(b, dtype=float)
        if a_arr.size == 0 or b_arr.size == 0:
            return 0.0
        a_flat = a_arr.ravel()
        b_flat = b_arr.ravel()
        dot = float(np.dot(a_flat, b_flat))
        norm_a = float(np.linalg.norm(a_flat))
        norm_b = float(np.linalg.norm(b_flat))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _split_into_paragraphs(self, text: str) -> list[str]:
        parts = PARAGRAPH_SPLIT.split(text.strip())
        return [p.strip() for p in parts if p.strip()]

    def _detect_heading(self, paragraph: str) -> str | None:
        heading_patterns = [
            re.compile(r"^(?:CHAPTER|Chapter|PART|Part|SECTION|Section)\s+\d+[.:]\s*(.+)", re.IGNORECASE),
            re.compile(r"^(?:[\d]+[.)]\s+)(.+)"),
            re.compile(r"^[A-Z][A-Z\s]{2,}$"),
        ]
        stripped = paragraph.strip()
        if len(stripped) > 3 and len(stripped) <= 120:
            if stripped.isupper() and len(stripped) >= 3:
                return stripped
            for pat in heading_patterns:
                m = pat.match(stripped)
                if m:
                    return stripped
        return None

    def _simple_split(self, text: str) -> list[str]:
        tokens = self._tokenize_text(text)
        chunks = []
        if len(tokens) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + self.chunk_size]
            if not chunk_tokens:
                continue
            chunk_text = " ".join(chunk_tokens)
            chunks.append(chunk_text)
            if i + self.chunk_size >= len(tokens):
                break
        return chunks

    def _semantic_split(self, text: str) -> list[str]:
        paragraphs = self._split_into_paragraphs(text)
        if len(paragraphs) <= 1:
            return self._simple_split(text)

        try:
            embedder = self._get_embedder()
            embeddings = list(embedder.embed(paragraphs))
        except Exception:
            return self._paragraph_aware_split(paragraphs)

        breakpoints = [0]
        for i in range(1, len(paragraphs)):
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            if sim < float(self.semantic_threshold):
                breakpoints.append(i)

        chunks = []
        for j in range(len(breakpoints)):
            start = breakpoints[j]
            end = breakpoints[j + 1] if j + 1 < len(breakpoints) else len(paragraphs)
            segment = "\n\n".join(paragraphs[start:end])

            tokens = self._tokenize_text(segment)
            if len(tokens) <= self.chunk_size:
                chunks.append(segment)
            else:
                step = self.chunk_size - self.chunk_overlap
                for k in range(0, len(tokens), step):
                    chunk_tokens = tokens[k:k + self.chunk_size]
                    if chunk_tokens:
                        chunks.append(" ".join(chunk_tokens))

        return chunks

    def _paragraph_aware_split(self, paragraphs: list[str]) -> list[str]:
        chunks = []
        current = ""
        current_tokens = 0

        for para in paragraphs:
            para_tokens = len(self._tokenize_text(para))

            if current_tokens + para_tokens <= self.chunk_size:
                current = f"{current}\n\n{para}" if current else para
                current_tokens += para_tokens
            else:
                if current:
                    chunks.append(current)
                if para_tokens >= self.chunk_size:
                    sub_chunks = self._simple_split(para)
                    chunks.extend(sub_chunks)
                    current = sub_chunks[-1] if sub_chunks else ""
                    current_tokens = len(self._tokenize_text(current))
                else:
                    current = para
                    current_tokens = para_tokens

        if current:
            chunks.append(current)

        return chunks

    def _generate_llm_metadata(self, chunks: list[str]) -> list[dict]:
        results = []
        for chunk_text in chunks:
            metadata = {
                "keywords": [],
                "topics": [],
                "doshas": [],
                "symptoms": [],
                "treatments": [],
            }
            text_lower = chunk_text.lower()

            dosha_map = {
                "vata": "vata", "vatha": "vata", "vaata": "vata",
                "pitta": "pitta", "pitha": "pitta",
                "kapha": "kapha", "kafa": "kapha", "kabha": "kapha",
            }
            symptom_patterns = [
                r"\b(pain|fever|cough|headache|nausea|swelling|inflammation|fatigue|weakness|indigestion|constipation|diarrhea|vomiting|rashes|itching|anxiety|insomnia|asthma|arthritis|diabetes|hypertension|obesity)\b",
                r"\b(ache|discomfort|burning|irritation|bloating|gas|acidity|congestion)\b",
            ]
            treatment_patterns = [
                r"\b(churna|taila|ghrita|kwatha|asava|arishta|bhasma|rasa|lepa|vati|guggulu|choornam|kashayam|decoction|powder|oil|ghee|paste|tablet|massage|abhyanga|panchakarma|vamana|virechana|basti|nasya|raktamokshana|shirodhara)\b",
                r"\b(treatment|therapy|remedy|cure|medicine|herb|herbal|ayurvedic|dosage|prescription|regimen|diet|lifestyle|yoga|pranayama|meditation)\b",
            ]

            for word, dosha in dosha_map.items():
                if re.search(rf"\b{word}\b", text_lower):
                    if dosha not in metadata["doshas"]:
                        metadata["doshas"].append(dosha)

            for pat in symptom_patterns:
                for match in re.finditer(pat, text_lower, re.IGNORECASE):
                    word = match.group(0).lower()
                    if word not in metadata["symptoms"]:
                        metadata["symptoms"].append(word)

            for pat in treatment_patterns:
                for match in re.finditer(pat, text_lower, re.IGNORECASE):
                    word = match.group(0).lower()
                    if word not in metadata["treatments"]:
                        metadata["treatments"].append(word)

            metadata["keywords"] = list(set(metadata["symptoms"] + metadata["treatments"] + metadata["doshas"]))[:20]
            metadata["topics"] = [t for t in ["symptom", "treatment", "diagnosis", "prognosis", "prevention", "lifestyle"]
                                 if t in text_lower]

            results.append(metadata)
        return results

    def chunk_documents(self, documents: list, db) -> list[dict]:
        all_chunks = []
        last_doc_id = None

        for doc_idx, doc in enumerate(documents):
            text = doc.text
            title = doc.title
            doc_id = str(uuid.uuid4())
            source_url = doc.source_url
            source_path = doc.source_path

            document_children = []

            document_chunk = create_chunk(
                "document", text,
                doc_id=doc_id,
                title=title,
                source_url=source_url or source_path,
                section_path=[title],
                paragraph_start=None,
                paragraph_end=None,
                prev_id=last_doc_id,
                next_id=None,
                parent_id=None,
                children_ids=document_children,
            )

            if last_doc_id:
                db.update_next_id(last_doc_id, document_chunk["chunk_id"])

            last_doc_id = document_chunk["chunk_id"]
            db.insert_chunk(document_chunk)

            if self.split_on_paragraphs:
                paragraphs = self._split_into_paragraphs(text)
            else:
                paragraphs = [text]

            if self.use_semantic and len(paragraphs) > 1:
                chunk_texts = self._semantic_split(text)
            elif len(paragraphs) > 1:
                chunk_texts = self._paragraph_aware_split(paragraphs)
            else:
                chunk_texts = self._simple_split(text)

            if self.llm_metadata:
                llm_metadata_list = self._generate_llm_metadata(chunk_texts)
            else:
                llm_metadata_list = [{}] * len(chunk_texts)

            headings = []
            if self.split_on_paragraphs:
                for para in paragraphs:
                    heading = self._detect_heading(para)
                    if heading:
                        headings.append(heading)

            document_children = []

            for chunk_idx, (chunk_text, meta) in enumerate(zip(chunk_texts, llm_metadata_list)):
                near_heading = headings[-1] if headings else None
                section_path = [title]
                if near_heading:
                    section_path.append(near_heading)

                prev_id = all_chunks[-1]["chunk_id"] if all_chunks else None

                chunk = create_chunk(
                    "chunk", chunk_text,
                    doc_id=doc_id,
                    title=title,
                    source_url=source_url or source_path,
                    section_path=section_path,
                    paragraph_start=chunk_idx,
                    paragraph_end=chunk_idx + 1,
                    prev_id=prev_id,
                    next_id=None,
                    parent_id=document_chunk["chunk_id"],
                    keywords=meta.get("keywords", []),
                    topics=meta.get("topics", []),
                    doshas=meta.get("doshas", []),
                    symptoms=meta.get("symptoms", []),
                    treatments=meta.get("treatments", []),
                    llm_metadata=meta,
                )

                if prev_id:
                    db.update_next_id(prev_id, chunk["chunk_id"])

                document_children.append(chunk["chunk_id"])
                all_chunks.append(chunk)
                db.insert_chunk(chunk)

            db.update_children_ids(document_chunk["chunk_id"], document_children)
            document_chunk["children_ids"] = document_children
            all_chunks.append(document_chunk)

            db.insert_document({
                "doc_id": doc_id,
                "title": title,
                "source_type": doc.source_type,
                "source_path": source_path or "",
                "source_url": source_url or "",
                "file_name": doc.file_name or "",
                "chunk_count": len(document_children),
                "metadata": doc.metadata,
            })

        return all_chunks


def chunk_page(page_data):
    raise NotImplementedError(
        "Legacy Wikipedia chunking has been removed. Use HybridChunker instead."
    )


def process_pages(pages, db, **kwargs):
    raise NotImplementedError(
        "Legacy Wikipedia processing has been removed. Use HybridChunker.chunk_documents() instead."
    )
