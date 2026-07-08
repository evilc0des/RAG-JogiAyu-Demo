import json
import sqlite3
from pathlib import Path


class ChunkStoreDB:
    def __init__(self, db_path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-64000")
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
                chunk_id        TEXT PRIMARY KEY,
                doc_id          TEXT,
                chunk_type      TEXT NOT NULL,
                text            TEXT NOT NULL,
                section_path    TEXT,
                title           TEXT,
                source_url      TEXT,
                paragraph_start INTEGER,
                paragraph_end   INTEGER,
                prev_id         TEXT,
                next_id         TEXT,
                parent_id       TEXT,
                children_ids    TEXT,
                keywords        TEXT,
                topics          TEXT,
                doshas          TEXT,
                symptoms        TEXT,
                treatments      TEXT,
                llm_metadata    TEXT
            )"""
        )
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS documents (
                doc_id       TEXT PRIMARY KEY,
                title        TEXT,
                source_type  TEXT,
                source_path  TEXT,
                source_url   TEXT,
                file_name    TEXT,
                chunk_count  INTEGER DEFAULT 0,
                created_at   TEXT DEFAULT (datetime('now')),
                metadata     TEXT
            )"""
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)")
        self.conn.commit()

    def insert_chunk(self, chunk):
        section_path = json.dumps(chunk.get("section_path")) if chunk.get("section_path") is not None else None
        children_ids = json.dumps(chunk.get("children_ids")) if chunk.get("children_ids") else "[]"
        keywords = json.dumps(chunk.get("keywords", []))
        topics = json.dumps(chunk.get("topics", []))
        doshas = json.dumps(chunk.get("doshas", []))
        symptoms = json.dumps(chunk.get("symptoms", []))
        treatments = json.dumps(chunk.get("treatments", []))
        llm_metadata = json.dumps(chunk.get("llm_metadata", {}))

        self.conn.execute(
            """INSERT OR REPLACE INTO chunks
               (chunk_id, doc_id, chunk_type, text, section_path, title, source_url,
                paragraph_start, paragraph_end, prev_id, next_id, parent_id, children_ids,
                keywords, topics, doshas, symptoms, treatments, llm_metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                chunk["chunk_id"],
                chunk.get("doc_id"),
                chunk.get("chunk_type"),
                chunk.get("text"),
                section_path,
                chunk.get("title"),
                chunk.get("source_url"),
                chunk.get("paragraph_start"),
                chunk.get("paragraph_end"),
                chunk.get("prev_id"),
                chunk.get("next_id"),
                chunk.get("parent_id"),
                children_ids,
                keywords,
                topics,
                doshas,
                symptoms,
                treatments,
                llm_metadata,
            ),
        )

    def insert_document(self, doc_record: dict):
        metadata_json = json.dumps(doc_record.get("metadata", {}))
        self.conn.execute(
            """INSERT OR REPLACE INTO documents
               (doc_id, title, source_type, source_path, source_url, file_name, chunk_count, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                doc_record.get("doc_id"),
                doc_record.get("title"),
                doc_record.get("source_type"),
                doc_record.get("source_path"),
                doc_record.get("source_url"),
                doc_record.get("file_name"),
                doc_record.get("chunk_count", 0),
                metadata_json,
            ),
        )

    def get_chunk(self, chunk_id):
        row = self.conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def get_children_by_type(self, chunk_type, limit=1000, offset=0):
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE chunk_type = ? ORDER BY rowid LIMIT ? OFFSET ?",
            (chunk_type, limit, offset),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_chunks_by_doc_id(self, doc_id, chunk_type=None):
        if chunk_type:
            rows = self.conn.execute(
                "SELECT * FROM chunks WHERE doc_id = ? AND chunk_type = ?",
                (str(doc_id), chunk_type),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM chunks WHERE doc_id = ?",
                (str(doc_id),),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count_children(self, chunk_type=None):
        if chunk_type:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE chunk_type = ?", (chunk_type,)
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0] if row else 0

    def commit(self):
        self.conn.commit()

    def update_next_id(self, chunk_id, next_id):
        self.conn.execute(
            "UPDATE chunks SET next_id = ? WHERE chunk_id = ?",
            (next_id, chunk_id),
        )

    def update_children_ids(self, chunk_id, children_ids):
        self.conn.execute(
            "UPDATE chunks SET children_ids = ? WHERE chunk_id = ?",
            (json.dumps(children_ids), chunk_id),
        )

    def get_last_chunk_id(self, chunk_type):
        row = self.conn.execute(
            "SELECT chunk_id FROM chunks WHERE chunk_type = ? ORDER BY rowid DESC LIMIT 1",
            (chunk_type,),
        ).fetchone()
        return row[0] if row else None

    def get_last_page_doc_id(self):
        row = self.conn.execute(
            "SELECT doc_id FROM chunks WHERE chunk_type = 'document' ORDER BY rowid DESC LIMIT 1",
        ).fetchone()
        return row[0] if row else None

    def get_document(self, doc_id):
        row = self.conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (str(doc_id),)
        ).fetchone()
        if row is None:
            return None
        keys = ["doc_id", "title", "source_type", "source_path", "source_url",
                "file_name", "chunk_count", "created_at", "metadata"]
        d = dict(zip(keys, row))
        if d.get("metadata"):
            try:
                d["metadata"] = json.loads(d["metadata"])
            except (json.JSONDecodeError, TypeError):
                d["metadata"] = {}
        else:
            d["metadata"] = {}
        return d

    def get_all_documents(self, limit=100, offset=0):
        rows = self.conn.execute(
            "SELECT * FROM documents ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        keys = ["doc_id", "title", "source_type", "source_path", "source_url",
                "file_name", "chunk_count", "created_at", "metadata"]
        return [dict(zip(keys, r)) for r in rows]

    def count_documents(self):
        row = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return row[0] if row else 0

    def close(self):
        self.conn.commit()
        self.conn.close()

    _CHUNK_KEYS = [
        "chunk_id", "doc_id", "chunk_type", "text", "section_path",
        "title", "source_url", "paragraph_start", "paragraph_end",
        "prev_id", "next_id", "parent_id", "children_ids",
        "keywords", "topics", "doshas", "symptoms", "treatments", "llm_metadata",
    ]

    def _row_to_dict(self, row):
        d = dict(zip(self._CHUNK_KEYS, row))
        json_fields = {
            "section_path": [],
            "children_ids": [],
            "keywords": [],
            "topics": [],
            "doshas": [],
            "symptoms": [],
            "treatments": [],
            "llm_metadata": {},
        }
        for field, default in json_fields.items():
            val = d.get(field)
            if val:
                try:
                    d[field] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    d[field] = default
            else:
                d[field] = default
        return d
