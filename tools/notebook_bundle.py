"""Build a readable, self-contained notebook with byte-preserving embedded modules."""
import base64
import hashlib
import json
from pathlib import Path
import zlib


class Notebook:
    def __init__(self):
        self.cells = []

    def add(self, kind, source):
        item = {"cell_type": kind, "metadata": {}, "source": source.strip() + "\n"}
        if kind == "code":
            compile(item["source"], "notebook cell", "exec")
            item.update(execution_count=None, outputs=[])
        self.cells.append(item)

    def embed(self, files, root, manifest_name):
        manifest = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        self.add("markdown", "## Code và dữ liệu nhãn nhúng\nChạy các cell tiếp theo để tạo module và CSV nguyên byte trong working. Không cần dataset code.")
        self.add("code", """
import hashlib, base64, zlib
def write_embedded(relative, payload, expected):
    assert hashlib.sha256(payload).hexdigest() == expected, f'Embedded file changed: {relative}'
    target = WORK / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
""")
        for path in files:
            if path.suffix != ".py":
                continue
            raw = path.read_bytes()
            source = raw.decode("utf-8").replace("\r\n", "\n")
            delimiter = "'" * 3
            assert delimiter not in source
            crlf = [i for i, line in enumerate(raw.split(b"\n")) if line.endswith(b"\r")]
            rel = path.relative_to(root).as_posix()
            cell = f"# {rel}\n_embedded_source = r{delimiter}{source}{delimiter}\n"
            cell += "_lines = _embedded_source.split('\\n')\n"
            cell += f"for _line_index in {crlf!r}:\n    _lines[_line_index] += '\\r'\n"
            cell += f"write_embedded({rel!r}, '\\n'.join(_lines).encode('utf-8'), {manifest[rel]!r})\n"
            self.add("code", cell)
        data = {p.relative_to(root).as_posix(): base64.b64encode(zlib.compress(p.read_bytes(), 9)).decode()
                for p in files if p.suffix != ".py"}
        self.add("code", "_embedded_data = " + repr(data) + "\n_manifest = " + repr(manifest) + "\n"
                 + "for relative, encoded in _embedded_data.items():\n"
                 + "    write_embedded(relative, zlib.decompress(base64.b64decode(encoded)), _manifest[relative])\n"
                 + f"(WORK / {manifest_name!r}).write_text(json.dumps(_manifest, indent=2), encoding='utf-8')\n"
                 + "print('Embedded files verified:', len(_manifest))\n")
        return manifest

    def write(self, path):
        notebook = {"cells": self.cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3", "language": "python"},
                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
        Path(path).write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
