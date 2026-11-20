"""Immutable upstream snapshots, with Git-blob SHA-1 verification.

Source identity is necessary but NOT sufficient for algorithmic fidelity.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/"docs"/"ORIGINAL_UPSTREAM_BLOB_MANIFEST.json"


def test_all_vendored_originals_have_bit_exact_pinned_git_blob_hashes():
    record=json.loads(MANIFEST.read_text())
    assert record["schema"]=="frozen_upstream_git_blobs_v1"
    assert set(record["repositories"])=={"glfc","lga","feat","fedclass"}
    expected={"glfc":17,"lga":16,"feat":18,"fedclass":25}
    for suite,repository in record["repositories"].items():
        assert repository["original_file_count"]==expected[suite]
        assert len(repository["files"])==expected[suite]
        for relative,item in repository["files"].items():
            path=ROOT/relative
            assert path.is_file(),f"Missing original upstream file: {path}"
            raw=path.read_bytes()
            header=b"blob "+str(len(raw)).encode()+b"\x00"
            assert hashlib.sha1(header+raw).hexdigest()==item["git_blob_sha1"],relative
            assert len(raw)==item["size"]
            actual_mode="100755" if path.stat().st_mode&0o111 else "100644"
            assert actual_mode==item["mode"],relative
    # The copied author source must not be confused with the adjacent
    # /pilot folders, which contain our own experimental adapters.
    assert sum(v["original_file_count"] for v in record["repositories"].values())==76
