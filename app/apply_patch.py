"""Parse and apply Codex / OpenAI-style ``apply_patch`` text locally.

Format (freeform or function ``input`` string)::

    *** Begin Patch
    *** Add File: path/to/new.txt
    +line one
    +line two
    *** Update File: src/app.ts
    @@
     context
    -old
    +new
    *** Delete File: obsolete.txt
    *** End Patch

Optional rename inside an update::

    *** Update File: old.py
    *** Move to: new.py
    @@
    -a
    +b

Paths are resolved under a configured workspace root; escapes are rejected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple, Union

ActionKind = Literal["add", "update", "delete"]


class ApplyPatchError(ValueError):
    """Invalid patch text or failed apply."""


@dataclass
class HunkLine:
    kind: Literal[" ", "-", "+"]
    text: str  # without the leading marker


@dataclass
class FileAction:
    kind: ActionKind
    path: str
    move_to: Optional[str] = None
    # For add: full new content lines (no markers)
    # For update: list of hunks (each hunk is list of HunkLine)
    hunks: List[List[HunkLine]] = field(default_factory=list)
    add_lines: List[str] = field(default_factory=list)


@dataclass
class ApplyResult:
    ok: bool
    message: str
    changed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def as_tool_output(self) -> str:
        parts = [self.message]
        if self.changed:
            parts.append("changed: " + ", ".join(self.changed))
        if self.errors:
            parts.append("errors: " + "; ".join(self.errors))
        return "\n".join(parts)


_BEGIN = re.compile(r"^\*\*\*\s*Begin Patch\s*$", re.I)
_END = re.compile(r"^\*\*\*\s*End Patch\s*$", re.I)
_ADD = re.compile(r"^\*\*\*\s*Add File:\s*(.+?)\s*$", re.I)
_UPDATE = re.compile(r"^\*\*\*\s*Update File:\s*(.+?)\s*$", re.I)
_DELETE = re.compile(r"^\*\*\*\s*Delete File:\s*(.+?)\s*$", re.I)
_MOVE = re.compile(r"^\*\*\*\s*Move to:\s*(.+?)\s*$", re.I)
_HUNK = re.compile(r"^@@")


def _strip_bom(text: str) -> str:
    if text.startswith("\ufeff"):
        return text[1:]
    return text


def extract_patch_text(arguments: Union[str, dict, None]) -> str:
    """Pull patch body from function-call arguments (JSON object or raw string)."""
    if arguments is None:
        return ""
    if isinstance(arguments, dict):
        for key in ("input", "patch", "content", "text"):
            v = arguments.get(key)
            if isinstance(v, str) and v.strip():
                return v
        # single string value
        if len(arguments) == 1:
            only = next(iter(arguments.values()))
            if isinstance(only, str):
                return only
        return ""
    s = str(arguments).strip()
    if s.startswith("{"):
        try:
            import json

            data = json.loads(s)
            if isinstance(data, dict):
                return extract_patch_text(data)
        except Exception:
            pass
    return str(arguments)


def parse_apply_patch(text: str) -> List[FileAction]:
    """Parse full patch document into file actions. Raises ApplyPatchError."""
    text = _strip_bom(text or "")
    if not text.strip():
        raise ApplyPatchError("empty patch")

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    # Allow missing Begin/End for a single action block (some clients)
    has_begin = any(_BEGIN.match(ln) for ln in lines)
    has_end = any(_END.match(ln) for ln in lines)

    i = 0
    if has_begin:
        while i < len(lines) and not _BEGIN.match(lines[i]):
            i += 1
        if i >= len(lines):
            raise ApplyPatchError("missing *** Begin Patch")
        i += 1
    end_at = len(lines)
    if has_end:
        for j in range(i, len(lines)):
            if _END.match(lines[j]):
                end_at = j
                break
        else:
            raise ApplyPatchError("missing *** End Patch")

    actions: List[FileAction] = []
    current: Optional[FileAction] = None
    mode: Optional[str] = None  # "add" | "hunk" | None
    hunk: List[HunkLine] = []

    def flush_hunk() -> None:
        nonlocal hunk
        if current is not None and hunk:
            current.hunks.append(hunk)
            hunk = []

    def flush_action() -> None:
        nonlocal current, mode
        flush_hunk()
        if current is not None:
            actions.append(current)
            current = None
        mode = None

    while i < end_at:
        ln = lines[i]
        i += 1

        if _ADD.match(ln):
            flush_action()
            path = _ADD.match(ln).group(1).strip()  # type: ignore[union-attr]
            current = FileAction(kind="add", path=path)
            mode = "add"
            continue
        if _UPDATE.match(ln):
            flush_action()
            path = _UPDATE.match(ln).group(1).strip()  # type: ignore[union-attr]
            current = FileAction(kind="update", path=path)
            mode = None
            continue
        if _DELETE.match(ln):
            flush_action()
            path = _DELETE.match(ln).group(1).strip()  # type: ignore[union-attr]
            current = FileAction(kind="delete", path=path)
            mode = None
            flush_action()
            continue
        if _MOVE.match(ln) and current and current.kind == "update":
            current.move_to = _MOVE.match(ln).group(1).strip()  # type: ignore[union-attr]
            continue
        if _HUNK.match(ln) and current and current.kind == "update":
            flush_hunk()
            mode = "hunk"
            continue
        if current is None:
            if not ln.strip():
                continue
            # ignore preamble noise
            continue

        if current.kind == "add" and mode == "add":
            if ln.startswith("+"):
                current.add_lines.append(ln[1:])
            elif ln.startswith("***"):
                # next action header — reprocess
                i -= 1
                flush_action()
            else:
                # bare line treated as content
                current.add_lines.append(ln)
            continue

        if current.kind == "update" and mode == "hunk":
            if ln.startswith("***"):
                i -= 1
                flush_action()
                continue
            if _HUNK.match(ln):
                flush_hunk()
                continue
            if not ln:
                # empty context line without marker — treat as context ""
                hunk.append(HunkLine(kind=" ", text=""))
                continue
            ch = ln[0]
            if ch in (" ", "-", "+"):
                hunk.append(HunkLine(kind=ch, text=ln[1:]))  # type: ignore[arg-type]
            else:
                # unprefixed = context
                hunk.append(HunkLine(kind=" ", text=ln))
            continue

        if ln.startswith("***"):
            i -= 1
            flush_action()
            continue

    flush_action()
    if not actions:
        raise ApplyPatchError("no file actions in patch")
    return actions


def resolve_under_root(root: Path, rel: str) -> Path:
    """Resolve path under root; reject escapes and absolute paths outside root."""
    root = root.resolve()
    raw = (rel or "").strip().replace("\\", "/")
    if not raw or raw in (".",):
        raise ApplyPatchError("empty path")
    # Disallow absolute Windows / POSIX that ignore root
    p = Path(raw)
    if p.is_absolute():
        target = p.resolve()
        try:
            target.relative_to(root)
        except ValueError as e:
            raise ApplyPatchError(f"path escapes workspace: {rel}") from e
        return target
    # strip leading ./
    while raw.startswith("./"):
        raw = raw[2:]
    if ".." in Path(raw).parts:
        # still resolve and check
        target = (root / raw).resolve()
        try:
            target.relative_to(root)
        except ValueError as e:
            raise ApplyPatchError(f"path escapes workspace: {rel}") from e
        return target
    return (root / raw).resolve()


def _apply_hunks(original: str, hunks: Sequence[Sequence[HunkLine]]) -> str:
    """Apply sequential search/replace hunks (context + diffs) to file text."""
    # Normalize to lines without trailing newlines preserved via splitlines keepends false
    lines = original.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    # If original ended with newline, split leaves trailing ""; fine for matching
    pos = 0
    for hunk in hunks:
        old_lines = [hl.text for hl in hunk if hl.kind in (" ", "-")]
        new_lines = [hl.text for hl in hunk if hl.kind in (" ", "+")]
        if not old_lines and not new_lines:
            continue
        # Find old_lines starting at or after pos
        idx = _find_subseq(lines, old_lines, pos)
        if idx < 0:
            # try from start once
            idx = _find_subseq(lines, old_lines, 0)
        if idx < 0:
            preview = "\n".join(old_lines[:5])
            raise ApplyPatchError(f"hunk context not found:\n{preview}")
        end = idx + len(old_lines)
        lines = list(lines[:idx]) + list(new_lines) + list(lines[end:])
        pos = idx + len(new_lines)
    return "\n".join(lines)


def _find_subseq(hay: Sequence[str], needle: Sequence[str], start: int) -> int:
    if not needle:
        return start
    n = len(needle)
    for i in range(start, len(hay) - n + 1):
        if list(hay[i : i + n]) == list(needle):
            return i
    return -1


def apply_actions(
    actions: Sequence[FileAction],
    *,
    root: Path,
    dry_run: bool = False,
) -> ApplyResult:
    """Apply parsed actions under root. Returns ApplyResult."""
    root = Path(root).resolve()
    if not dry_run:
        root.mkdir(parents=True, exist_ok=True)

    changed: List[str] = []
    errors: List[str] = []

    for act in actions:
        try:
            target = resolve_under_root(root, act.path)
            if act.kind == "delete":
                if not dry_run:
                    if target.is_file():
                        target.unlink()
                    elif target.exists():
                        raise ApplyPatchError(f"not a file: {act.path}")
                changed.append(f"delete {act.path}")
                continue

            if act.kind == "add":
                content = "\n".join(act.add_lines)
                if act.add_lines and not content.endswith("\n"):
                    # match common patch style: last line may omit final NL;
                    # write as joined lines + trailing newline if original had + lines
                    content = content + "\n" if act.add_lines else content
                if not dry_run:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        raise ApplyPatchError(f"add failed, exists: {act.path}")
                    target.write_text(content, encoding="utf-8")
                changed.append(f"add {act.path}")
                continue

            # update
            if not target.is_file():
                raise ApplyPatchError(f"update missing file: {act.path}")
            original = target.read_text(encoding="utf-8")
            new_text = _apply_hunks(original, act.hunks) if act.hunks else original

            dest = target
            rel_out = act.path
            if act.move_to:
                dest = resolve_under_root(root, act.move_to)
                rel_out = f"{act.path} -> {act.move_to}"

            if not dry_run:
                if act.move_to:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if dest.exists() and dest != target:
                        raise ApplyPatchError(f"move target exists: {act.move_to}")
                    target.unlink()
                dest.write_text(new_text, encoding="utf-8")
            changed.append(f"update {rel_out}")
        except ApplyPatchError as e:
            errors.append(str(e))
        except OSError as e:
            errors.append(f"{act.path}: {e}")

    ok = not errors
    if ok:
        msg = f"apply_patch ok ({len(changed)} ops)"
    else:
        msg = f"apply_patch partial/fail ({len(changed)} ok, {len(errors)} err)"
    return ApplyResult(ok=ok, message=msg, changed=changed, errors=errors)


def apply_patch_text(
    text: str,
    *,
    root: Path,
    dry_run: bool = False,
) -> ApplyResult:
    """Parse + apply. On parse error returns failed ApplyResult (no raise)."""
    try:
        actions = parse_apply_patch(text)
    except ApplyPatchError as e:
        return ApplyResult(ok=False, message=f"parse error: {e}", errors=[str(e)])
    return apply_actions(actions, root=root, dry_run=dry_run)


# ---------------------------------------------------------------------------
# Tools list normalization for xAI (custom → function; keep apply_patch)
# ---------------------------------------------------------------------------

APPLY_PATCH_FUNCTION_SCHEMA = {
    "type": "object",
    "properties": {
        "input": {
            "type": "string",
            "description": (
                "Full apply_patch document including "
                "*** Begin Patch and *** End Patch"
            ),
        }
    },
    "required": ["input"],
}


def _tool_name(t: dict) -> str:
    if isinstance(t.get("function"), dict):
        return str(t["function"].get("name") or "")
    return str(t.get("name") or "")


def normalize_tools_for_xai(
    tools: Optional[Sequence[dict]],
    *,
    strip_apply_patch: bool = False,
) -> Tuple[List[dict], bool]:
    """Normalize tools for xAI function/built-in only.

    - ``type: custom`` → ``type: function`` (flat Responses style)
    - ensure function tools have ``parameters``
    - optionally strip ``apply_patch`` (CPA-like); default **keep** as function

    Returns (tools, saw_apply_patch).
    """
    if not tools:
        return [], False
    out: List[dict] = []
    saw = False
    for t in tools:
        if not isinstance(t, dict):
            continue
        t = dict(t)
        name = _tool_name(t)
        if name == "apply_patch":
            saw = True
            if strip_apply_patch:
                continue
            # Force function form xAI accepts
            out.append(
                {
                    "type": "function",
                    "name": "apply_patch",
                    "description": t.get("description")
                    or (
                        "Apply a unified multi-file patch. "
                        "Argument `input` is the full *** Begin Patch … *** End Patch text."
                    ),
                    "parameters": (
                        t.get("parameters")
                        if isinstance(t.get("parameters"), dict)
                        else (
                            (t.get("function") or {}).get("parameters")
                            if isinstance(t.get("function"), dict)
                            else None
                        )
                    )
                    or APPLY_PATCH_FUNCTION_SCHEMA,
                }
            )
            continue

        typ = (t.get("type") or "function").strip()
        if typ == "custom":
            # custom → function
            out.append(
                {
                    "type": "function",
                    "name": t.get("name") or name or "tool",
                    "description": t.get("description") or "",
                    "parameters": t.get("parameters")
                    or {"type": "object", "properties": {}},
                }
            )
            continue
        if typ == "function":
            if isinstance(t.get("function"), dict):
                # Chat nested form → flat Responses form for official wire
                fn = t["function"]
                item = {
                    "type": "function",
                    "name": fn.get("name") or "",
                    "description": fn.get("description") or "",
                    "parameters": fn.get("parameters")
                    or {"type": "object", "properties": {}},
                }
                out.append(item)
            else:
                if "parameters" not in t:
                    t["parameters"] = {"type": "object", "properties": {}}
                t["type"] = "function"
                out.append(t)
            continue
        # built-ins / unknown — pass through
        out.append(t)
    return out, saw


def is_apply_patch_call(name: Optional[str]) -> bool:
    return (name or "").strip() == "apply_patch"


def tools_to_chat_nested(tools: Sequence[dict]) -> List[dict]:
    """Convert flat Responses-style function tools to Chat Completions nested form."""
    out: List[dict] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            out.append(dict(t))
            continue
        if (t.get("type") or "function") == "function" and t.get("name"):
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.get("name") or "",
                        "description": t.get("description") or "",
                        "parameters": t.get("parameters")
                        or {"type": "object", "properties": {}},
                    },
                }
            )
            continue
        out.append(dict(t))
    return out


def iter_apply_patch_payloads_from_chat(data: dict) -> List[str]:
    """Extract apply_patch argument strings from a Chat Completions response."""
    payloads: List[str] = []
    if not isinstance(data, dict):
        return payloads
    for choice in data.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        msg = choice.get("message") or {}
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            if not is_apply_patch_call(fn.get("name") or tc.get("name")):
                continue
            payloads.append(extract_patch_text(fn.get("arguments")))
    return payloads


def iter_apply_patch_payloads_from_responses(data: dict) -> List[str]:
    """Extract apply_patch argument strings from a Responses completed object."""
    payloads: List[str] = []
    if not isinstance(data, dict):
        return payloads
    if data.get("object") != "response" and isinstance(data.get("response"), dict):
        data = data["response"]
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function_call":
            continue
        if not is_apply_patch_call(item.get("name")):
            continue
        args = item.get("arguments") or ""
        payloads.append(extract_patch_text(args))
    return payloads


def maybe_local_apply_from_response(
    data: dict,
    *,
    protocol: Literal["chat", "responses"],
    root: Path,
    enabled: bool,
) -> Optional[ApplyResult]:
    """If enabled, apply all apply_patch calls found in a non-stream response."""
    if not enabled or not isinstance(data, dict):
        return None
    if protocol == "chat":
        payloads = iter_apply_patch_payloads_from_chat(data)
    else:
        payloads = iter_apply_patch_payloads_from_responses(data)
    if not payloads:
        return None
    combined = ApplyResult(ok=True, message="apply_patch local", changed=[], errors=[])
    for text in payloads:
        if not (text or "").strip():
            combined.errors.append("empty apply_patch arguments")
            combined.ok = False
            continue
        r = apply_patch_text(text, root=root)
        combined.changed.extend(r.changed)
        combined.errors.extend(r.errors)
        if not r.ok:
            combined.ok = False
    if combined.ok:
        combined.message = f"apply_patch local ok ({len(combined.changed)} ops)"
    else:
        combined.message = (
            f"apply_patch local fail ({len(combined.changed)} ok, "
            f"{len(combined.errors)} err)"
        )
    return combined
