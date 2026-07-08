from dataclasses import dataclass, field
from pathlib import Path
import json
import csv


@dataclass
class ParsedDocument:
    title: str
    text: str
    source_type: str
    source_path: str | None = None
    source_url: str | None = None
    file_name: str | None = None
    metadata: dict = field(default_factory=dict)


SUPPORTED_EXTENSIONS = {
    ".txt": "text",
    ".md": "text",
    ".json": "transcript",
    ".csv": "transcript",
    ".log": "text",
    ".pdf": "pdf",
}


def read_text_file(file_path: Path) -> ParsedDocument:
    text = file_path.read_text(encoding="utf-8")
    return ParsedDocument(
        title=file_path.stem,
        text=text,
        source_type="text",
        source_path=str(file_path),
        file_name=file_path.name,
    )


def read_json_transcript(file_path: Path) -> ParsedDocument:
    data = json.loads(file_path.read_text(encoding="utf-8"))
    title = data.get("title", file_path.stem)
    text = _extract_text_from_json(data, file_path.stem)
    metadata = {k: v for k, v in data.items() if k not in ("title", "text", "transcript", "segments", "content")}
    return ParsedDocument(
        title=title,
        text=text,
        source_type="transcript",
        source_path=str(file_path),
        file_name=file_path.name,
        metadata=metadata,
    )


def read_csv_transcript(file_path: Path) -> ParsedDocument:
    rows = []
    with open(file_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    text_parts = []
    for row in rows:
        segment_text = row.get("text") or row.get("content") or row.get("transcript") or ""
        speaker = row.get("speaker") or row.get("author") or ""
        timestamp = row.get("timestamp") or row.get("start") or ""
        prefix = f"[{timestamp}] " if timestamp else ""
        speaker_prefix = f"{speaker}: " if speaker else ""
        if segment_text.strip():
            text_parts.append(f"{prefix}{speaker_prefix}{segment_text.strip()}")

    return ParsedDocument(
        title=file_path.stem,
        text="\n\n".join(text_parts),
        source_type="transcript",
        source_path=str(file_path),
        file_name=file_path.name,
    )


def read_pdf_file(file_path: Path) -> ParsedDocument:
    from pypdf import PdfReader
    reader = PdfReader(str(file_path))
    pages = []
    metadata = {}
    if reader.metadata:
        metadata = {k[1:] if k.startswith("/") else k: str(v)
                    for k, v in (reader.metadata or {}).items() if v}
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            pages.append(text.strip())
    full_text = "\n\f\n".join(pages)
    return ParsedDocument(
        title=file_path.stem,
        text=full_text,
        source_type="pdf",
        source_path=str(file_path),
        file_name=file_path.name,
        metadata=metadata,
    )


def read_raw_text(text: str, title: str = "Pasted Text") -> ParsedDocument:
    return ParsedDocument(
        title=title,
        text=text,
        source_type="text",
        metadata={"source": "manual_input"},
    )


def _extract_text_from_json(data: dict, default_title: str) -> str:
    for key in ("text", "transcript", "content", "body"):
        if key in data:
            return data[key]

    segments = data.get("segments")
    if isinstance(segments, list):
        parts = []
        for seg in segments:
            if isinstance(seg, dict):
                t = seg.get("text") or seg.get("content") or ""
                if t.strip():
                    parts.append(t.strip())
            elif isinstance(seg, str):
                if seg.strip():
                    parts.append(seg.strip())
        if parts:
            return "\n\n".join(parts)

    return json.dumps(data, indent=2)


def read_file(file_path: Path | str) -> ParsedDocument:
    path = Path(file_path) if isinstance(file_path, str) else file_path
    ext = path.suffix.lower()

    if ext == ".json":
        return read_json_transcript(path)
    elif ext == ".csv":
        return read_csv_transcript(path)
    elif ext == ".pdf":
        return read_pdf_file(path)
    elif ext in (".txt", ".md", ".log", ""):
        return read_text_file(path)
    else:
        return read_text_file(path)


def read_directory(dir_path: Path | str, extensions: set[str] | None = None) -> list[ParsedDocument]:
    path = Path(dir_path) if isinstance(dir_path, str) else dir_path
    if not path.is_dir():
        return []

    if extensions is None:
        extensions = set(SUPPORTED_EXTENSIONS.keys())

    docs = []
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            docs.append(read_file(file_path))
    return docs
