"""Shared fixtures: generate the sample PDFs once for the whole test session."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def sample_pdfs(tmp_path_factory):
    out = tmp_path_factory.mktemp("sample_pdfs")
    gen = ROOT / "sample_docs" / "generate_samples.py"
    # Patch the output dir by copying the script's constants: simplest is to
    # run the generator and copy the resulting PDFs.
    subprocess.run([sys.executable, str(gen)], check=True, capture_output=True)
    pdfs = sorted((ROOT / "sample_docs").glob("*.pdf"))
    assert len(pdfs) == 3, f"expected 3 sample PDFs, found {len(pdfs)}"
    for p in pdfs:
        (out / p.name).write_bytes(p.read_bytes())
    return [out / p.name for p in pdfs]
