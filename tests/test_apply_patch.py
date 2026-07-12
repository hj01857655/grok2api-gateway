"""Unit tests for local apply_patch parser + disk apply + tool normalize."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.apply_patch import (
    ApplyPatchError,
    apply_patch_text,
    extract_patch_text,
    is_apply_patch_call,
    normalize_tools_for_xai,
    parse_apply_patch,
    resolve_under_root,
)


SAMPLE = """\
*** Begin Patch
*** Add File: hello.txt
+hello
+world
*** Update File: a.txt
@@
 keep
-old
+new
*** Delete File: gone.txt
*** End Patch
"""


def test_parse_add_update_delete():
    actions = parse_apply_patch(SAMPLE)
    kinds = [a.kind for a in actions]
    assert kinds == ["add", "update", "delete"]
    assert actions[0].path == "hello.txt"
    assert actions[0].add_lines == ["hello", "world"]
    assert actions[1].path == "a.txt"
    assert len(actions[1].hunks) == 1
    assert actions[2].path == "gone.txt"


def test_apply_add_update_delete(tmp_path: Path):
    (tmp_path / "a.txt").write_text("keep\nold\n", encoding="utf-8")
    (tmp_path / "gone.txt").write_text("x\n", encoding="utf-8")

    result = apply_patch_text(SAMPLE, root=tmp_path)
    assert result.ok, result.as_tool_output()
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "hello\nworld\n"
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "keep\nnew\n"
    assert not (tmp_path / "gone.txt").exists()


def test_apply_move_to(tmp_path: Path):
    (tmp_path / "old.py").write_text("a\n", encoding="utf-8")
    patch = """\
*** Begin Patch
*** Update File: old.py
*** Move to: new.py
@@
-a
+b
*** End Patch
"""
    result = apply_patch_text(patch, root=tmp_path)
    assert result.ok, result.as_tool_output()
    assert not (tmp_path / "old.py").exists()
    assert (tmp_path / "new.py").read_text(encoding="utf-8") == "b\n"


def test_path_escape_rejected(tmp_path: Path):
    with pytest.raises(ApplyPatchError):
        resolve_under_root(tmp_path, "../outside.txt")


def test_hunk_context_missing(tmp_path: Path):
    (tmp_path / "f.txt").write_text("only\n", encoding="utf-8")
    patch = """\
*** Begin Patch
*** Update File: f.txt
@@
-missing
+x
*** End Patch
"""
    result = apply_patch_text(patch, root=tmp_path)
    assert not result.ok
    assert result.errors


def test_extract_patch_text_from_json_args():
    assert "Begin Patch" in extract_patch_text(
        {"input": "*** Begin Patch\n*** End Patch"}
    )
    assert extract_patch_text('{"input":"*** Begin Patch\\n*** End Patch"}').startswith(
        "*** Begin"
    )


def test_normalize_tools_custom_to_function_keeps_apply_patch():
    tools = [
        {"type": "custom", "name": "apply_patch", "description": "patch files"},
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "q",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {"type": "custom", "name": "other", "description": "x"},
    ]
    out, saw = normalize_tools_for_xai(tools, strip_apply_patch=False)
    assert saw is True
    names = [t.get("name") for t in out]
    assert "apply_patch" in names
    assert "lookup" in names
    assert "other" in names
    ap = next(t for t in out if t.get("name") == "apply_patch")
    assert ap["type"] == "function"
    assert "parameters" in ap

    stripped, saw2 = normalize_tools_for_xai(tools, strip_apply_patch=True)
    assert saw2 is True
    assert all(t.get("name") != "apply_patch" for t in stripped)


def test_is_apply_patch_call():
    assert is_apply_patch_call("apply_patch")
    assert not is_apply_patch_call("edit_file")
