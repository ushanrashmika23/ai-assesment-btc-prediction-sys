"""
Run a project command and save its terminal output as a readable Markdown log.

    python src/runlog.py --name train -- python src/train.py --arch gru
    python src/runlog.py --name evaluate --split both

Why this exists
---------------
The numbers this project reports are only worth anything if the run that
produced them can be inspected. A log written by hand is a claim; a log written
by the command that produced it is evidence. This wrapper captures stdout and
stderr verbatim, strips the TensorFlow startup noise that says nothing about the
result, and writes `logs/<name>.md` with the exact command, the exit code and
the environment that produced it.

WHAT IS RECORDED, AND WHY IT IS NOT A LEAK
    The log is written AFTER the command finishes and is never read back by any
    other script. It is a record of what happened, not an input to what happens
    next. Deleting logs/ changes no result.

Usage
    python src/runlog.py --name <slug> -- <command> [args...]
    python src/runlog.py --name <slug> --title "..." -- <command> [args...]

The `--` separator is required: everything before it configures this wrapper,
everything after it is the command to run.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOGS_DIR = ROOT / "logs"

# TensorFlow prints a wall of environment noise on every start. None of it is
# about this project's results, and leaving it in buries the numbers. These
# patterns match ONLY that noise - a real traceback still reaches the log.
NOISE = [
    re.compile(r"^WARNING: All log messages before absl::InitializeLog"),
    re.compile(r"^I0000 \d\d:\d\d:\d\d"),                  # absl INFO lines
    re.compile(r"^WARNING:tensorflow:TensorFlow GPU support is not available"),
    re.compile(r"^To enable the following instructions:"),
    re.compile(r"^\s*$"),                                   # runs of blank lines
]


def _emit(text: str) -> None:
    """
    Echo a captured line to the terminal without dying on its encoding.

    The dashboard draws its frames with box-drawing characters. A Windows
    console whose codepage is cp1252 cannot encode those, and a bare print()
    raises UnicodeEncodeError - AFTER the wrapped command has already run, so
    the log it was about to write is lost. Display must never be able to
    destroy the capture, so an unencodable character degrades to `?` instead.
    """
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        sys.stdout.write(text.encode(enc, "replace").decode(enc, "replace"))
        sys.stdout.flush()


def strip_noise(text: str) -> str:
    """Drop TF startup chatter and collapse blank-line runs."""
    kept, previous_blank = [], False
    for line in text.splitlines():
        if any(p.search(line) for p in NOISE[:-1]):
            continue
        blank = not line.strip()
        if blank and previous_blank:
            continue
        previous_blank = blank
        kept.append(line)
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)


def md_fence(text: str) -> str:
    """
    Wrap `text` in a fence long enough to survive any backticks inside it.

    The logs contain Markdown tables and ASCII bars, but a future command could
    print anything; sizing the fence to the content is what stops the output
    from escaping its block and corrupting the rest of the file.
    """
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}\n{text}\n{fence}"


def git_revision() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=ROOT, capture_output=True, text=True,
                             timeout=10)
        return out.stdout.strip() or "not a git repository"
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def write_log(out_path: Path, title: str, command: str, returncode: int,
              started: datetime, finished: datetime, body: str,
              note: str | None, provenance: str) -> None:
    header = [
        f"# {title}",
        "",
        "| | |",
        "|---|---|",
        f"| command | `{command}` |",
        f"| exit code | `{returncode}` |",
        f"| started (UTC) | {started:%Y-%m-%d %H:%M:%S} |",
        f"| finished (UTC) | {finished:%Y-%m-%d %H:%M:%S} |",
        f"| wall clock | {(finished - started).total_seconds():.1f}s |",
        f"| python | {platform.python_version()} |",
        f"| platform | {platform.platform()} |",
        f"| cwd | `{ROOT}` |",
        f"| git | `{git_revision()}` |",
        f"| provenance | {provenance} |",
        "",
    ]
    if note:
        header += [note, ""]
    header += [
        "Output below is verbatim stdout+stderr from the command above, with "
        "only the",
        "TensorFlow startup banner removed. Reproduce it by running the "
        "command shown.",
        "",
        "## Output",
        "",
        md_fence(body),
        "",
    ]
    out_path.write_text("\n".join(header), encoding="utf-8")


def main() -> int:
    argv = sys.argv[1:]
    if "--" in argv:
        split = argv.index("--")
        head, command = argv[:split], argv[split + 1:]
        if not command:
            raise SystemExit("nothing to run after `--`")
    else:
        head, command = argv, []

    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--name", required=True,
                   help="log filename without extension, e.g. train -> "
                        "logs/train.md")
    p.add_argument("--title", default=None,
                   help="human title (default: the slug, title-cased)")
    p.add_argument("--note", default=None,
                   help="one line explaining what this run is for")
    p.add_argument("--from-file", default=None, metavar="PATH",
                   help="capture an ALREADY-WRITTEN log instead of running a "
                        "command. Use for a run too expensive to repeat "
                        "(a multi-hour training run). The header records that "
                        "the text came from this file rather than from this "
                        "invocation, so the log never overstates itself.")
    p.add_argument("--ran-command", default=None, metavar="CMD",
                   help="with --from-file: the command that actually produced "
                        "that file")
    p.add_argument("--ran-between", default=None, metavar="START,END",
                   help="with --from-file: when that run started and finished, "
                        "as ISO-8601 UTC timestamps")
    args = p.parse_args(head)

    LOGS_DIR.mkdir(exist_ok=True)
    out_path = LOGS_DIR / f"{args.name}.md"
    title = args.title or args.name.replace("_", " ").title()

    # -- capture an existing log rather than running anything --------------
    if args.from_file:
        src = Path(args.from_file)
        if not src.is_absolute():
            src = ROOT / src
        if not src.exists():
            raise SystemExit(f"no log at {src}")
        raw = src.read_text(encoding="utf-8", errors="replace")
        # A training run that died would have left a traceback in the file;
        # record that honestly rather than stamping every log "exit 0".
        crashed = "Traceback (most recent call last)" in raw
        started = finished = datetime.now(timezone.utc)
        if args.ran_between and "," in args.ran_between:
            a, b = args.ran_between.split(",", 1)
            started = datetime.fromisoformat(a.strip())
            finished = datetime.fromisoformat(b.strip())
        write_log(out_path, title, args.ran_command or f"(see provenance)",
                  1 if crashed else 0, started, finished, strip_noise(raw),
                  args.note,
                  f"captured from `{src.relative_to(ROOT)}` (written by an "
                  f"earlier invocation of the command shown, not by this one)")
        print(f"wrote {out_path.relative_to(ROOT)}  "
              f"({len(raw.splitlines())} lines captured"
              f"{', TRACEBACK PRESENT' if crashed else ''})")
        return 1 if crashed else 0

    if not command:
        raise SystemExit(
            "usage: python src/runlog.py --name <slug> -- <command> [args...]\n"
            "   or: python src/runlog.py --name <slug> --from-file <path>\n"
            "The `--` separator is required.")

    # Prefer UTF-8 for our own output so box-drawing survives on a console
    # that can show it; _emit still guards the ones that cannot.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                         # noqa: BLE001
        pass                          # a redirected/odd stdout: _emit covers it

    started = datetime.now(timezone.utc)
    print(f"$ {' '.join(command)}", flush=True)
    print("-" * 72, flush=True)

    # Stream to the terminal AND capture, so a long run is watchable live and
    # still lands in the log even if it is killed midway.
    #
    # stderr is folded into stdout because TensorFlow writes its progress
    # bars to stderr: capturing them separately and concatenating at the end
    # would put every epoch's line after the summary that explains it.
    chunks: list[str] = []
    proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        chunks.append(line)
        _emit(line)                   # encoding-safe: see _emit
    proc.wait()
    body = strip_noise("".join(chunks))
    finished = datetime.now(timezone.utc)
    print("-" * 72)
    print(f"exit {proc.returncode}   "
          f"{(finished - started).total_seconds():.1f}s")

    write_log(out_path, title, " ".join(command), proc.returncode, started,
              finished, body, args.note,
              "captured live by this invocation")
    print(f"wrote {out_path.relative_to(ROOT)}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
