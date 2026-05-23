"""MultiCoNER v2 loader.

Source: https://huggingface.co/datasets/MultiCoNER/multiconer_v2
Format: per-language CoNLL files, e.g. EN-English/en_test.conll
  - sentence header: `# id <uuid>`
  - each non-header line: `token _ _ BIO-tag`
  - blank line separates sentences

We reconstruct each sentence by joining tokens with single spaces, then
convert BIO spans (B-XXX … I-XXX) to character-level (start, end) ranges
into the joined text. Labels are kept verbatim (33 fine-grained classes
plus their B-/I- prefixes collapsed to the class name).

Languages available in the repo (folder name -> code):
  BN-Bangla / DE-German / EN-English / ES-Spanish / FA-Farsi /
  FR-French / HI-Hindi / IT-Italian / MULTI-Multilingual / PT-Portuguese
  / SV-Swedish / UK-Ukrainian / ZH-Chinese
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from huggingface_hub import hf_hub_download

from benchmarks.base import Benchmark, Document, PHISpan


_LANG_TO_FOLDER = {
    "en": ("EN-English", "en"),
    "de": ("DE-German", "de"),
    "es": ("ES-Spanish", "es"),
    "fr": ("FR-French", "fr"),
    "it": ("IT-Italian", "it"),
    "pt": ("PT-Portuguese", "pt"),
    "zh": ("ZH-Chinese", "zh"),
    "hi": ("HI-Hindi", "hi"),
    "bn": ("BN-Bangla", "bn"),
    "fa": ("FA-Farsi", "fa"),
    "sv": ("SV-Swedish", "sv"),
    "uk": ("UK-Ukrainian", "uk"),
}


class MultiCoNERv2(Benchmark):
    name = "multiconer_v2"
    language = "en"

    def __init__(self, root: Path, split: str = "test", language: str = "en"):
        super().__init__(root)
        self.split = split  # "train" / "dev" / "test"
        self.lang_code = language
        if language not in _LANG_TO_FOLDER:
            raise ValueError(f"unknown language {language!r}; choose from {list(_LANG_TO_FOLDER)}")
        self.language = language
        self._docs: list[Document] | None = None

    def _ensure_file(self) -> Path:
        folder, prefix = _LANG_TO_FOLDER[self.lang_code]
        rel = f"{folder}/{prefix}_{self.split}.conll"
        local_dir = self.root / "raw"
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / rel
        if not local_path.exists():
            hf_hub_download(
                "MultiCoNER/multiconer_v2",
                rel,
                repo_type="dataset",
                local_dir=str(local_dir),
            )
        return local_path

    def _load(self) -> list[Document]:
        path = self._ensure_file()
        docs: list[Document] = []
        with path.open(encoding="utf-8") as f:
            doc_id, tokens, tags = None, [], []
            for raw in f:
                line = raw.rstrip("\n")
                if line.startswith("# id "):
                    if doc_id is not None:
                        docs.append(_assemble(doc_id, tokens, tags, self.lang_code, self.split))
                    doc_id = line[5:].strip()
                    tokens, tags = [], []
                elif not line.strip():
                    # sentence separator within same doc (rare here)
                    continue
                else:
                    parts = line.split()
                    if len(parts) >= 4:
                        tokens.append(parts[0])
                        tags.append(parts[-1])
            if doc_id is not None and tokens:
                docs.append(_assemble(doc_id, tokens, tags, self.lang_code, self.split))
        return docs

    def __iter__(self) -> Iterator[Document]:
        if self._docs is None:
            self._docs = self._load()
        yield from self._docs

    def __len__(self) -> int:
        if self._docs is None:
            self._docs = self._load()
        return len(self._docs)


def _assemble(doc_id: str, tokens: list[str], tags: list[str], lang: str, split: str) -> Document:
    text = " ".join(tokens)
    # Pre-compute each token's char start in the joined text.
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for i, tok in enumerate(tokens):
        if i > 0:
            cursor += 1  # the join space
        offsets.append((cursor, cursor + len(tok)))
        cursor += len(tok)

    spans: list[PHISpan] = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag.startswith("B-"):
            label = tag[2:]
            start = offsets[i][0]
            end = offsets[i][1]
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{label}":
                end = offsets[j][1]
                j += 1
            spans.append(PHISpan(start=start, end=end, label=label, text=text[start:end]))
            i = j
        else:
            i += 1

    return Document(
        doc_id=f"mc2_{lang}_{split}_{doc_id}",
        text=text,
        gold_spans=spans,
        metadata={"language": lang, "split": split},
    )
