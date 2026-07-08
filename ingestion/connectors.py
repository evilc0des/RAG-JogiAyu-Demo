from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from .file_readers import ParsedDocument, read_file, read_directory, read_raw_text, SUPPORTED_EXTENSIONS


@dataclass
class IngestionJob:
    job_id: str
    status: str
    documents: list[ParsedDocument] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    progress: float = 0.0
    message: str = ""


class BaseConnector(ABC):
    @abstractmethod
    def ingest(self) -> list[ParsedDocument]:
        ...


class FileConnector(BaseConnector):
    def __init__(self, file_path: Path | str):
        self.file_path = Path(file_path) if isinstance(file_path, str) else file_path
        self._errors: list[str] = []

    def ingest(self) -> list[ParsedDocument]:
        if not self.file_path.exists():
            self._errors.append(f"File not found: {self.file_path}")
            return []
        try:
            doc = read_file(self.file_path)
            return [doc]
        except Exception as e:
            self._errors.append(f"Error reading {self.file_path.name}: {e}")
            return []

    @property
    def errors(self) -> list[str]:
        return self._errors


class DirectoryConnector(BaseConnector):
    def __init__(self, dir_path: Path | str, extensions: set[str] | None = None):
        self.dir_path = Path(dir_path) if isinstance(dir_path, str) else dir_path
        self.extensions = extensions or set(SUPPORTED_EXTENSIONS.keys())
        self._errors: list[str] = []

    def ingest(self) -> list[ParsedDocument]:
        if not self.dir_path.exists():
            self._errors.append(f"Directory not found: {self.dir_path}")
            return []
        try:
            docs = read_directory(self.dir_path, self.extensions)
            if not docs:
                self._errors.append(f"No supported files found in {self.dir_path}")
            return docs
        except Exception as e:
            self._errors.append(f"Error reading directory {self.dir_path}: {e}")
            return []

    @property
    def errors(self) -> list[str]:
        return self._errors


class RawTextConnector(BaseConnector):
    def __init__(self, text: str, title: str = "Pasted Text"):
        self.text = text
        self.title = title

    def ingest(self) -> list[ParsedDocument]:
        return [read_raw_text(self.text, self.title)]


class WebUploadConnector(BaseConnector):
    def __init__(self, files: list[tuple[str, bytes, str]]):
        self.files = files
        self._errors: list[str] = []

    def ingest(self) -> list[ParsedDocument]:
        import tempfile
        docs = []
        for filename, content, _mime_type in self.files:
            ext = Path(filename).suffix.lower()
            temp_path = Path(tempfile.mkdtemp()) / filename
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_bytes(content)
            try:
                doc = read_file(temp_path)
                doc.file_name = filename
                docs.append(doc)
            except Exception as e:
                self._errors.append(f"Error reading {filename}: {e}")
        return docs

    @property
    def errors(self) -> list[str]:
        return self._errors
