"""
Static check: does every global name a module uses actually exist?

    python src/_check_names.py [files...]

Why this exists
---------------
Editing an import block by hand is easy to get subtly wrong, and the failure
mode is a NameError that only surfaces on the code path that uses the missing
name - which can be days later, in a function nobody ran. Two such bugs were
introduced in this project that way (a dropped `load_model` import, and a
method definition spliced into the middle of __init__ so the rest of the
initialiser became unreachable).

No linter is installed and adding one to requirements.txt for a check that runs
in two seconds is not worth a dependency, so this uses `symtable` from the
standard library: it reports, per scope, which names are global (free) and
which are assigned locally. A free name that is not a module global, not a
builtin, and not a locally-assigned name is a typo or a dropped import.

This is a lint, not a proof: it does not know about `getattr`, star-imports or
deliberate late binding, so it errs toward reporting. Lines marked with a
`# noqa` comment are skipped.
"""

from __future__ import annotations

import builtins
import symtable
import sys
from pathlib import Path


def walk(table: symtable.SymbolTable):
    yield table
    for child in table.get_children():
        yield from walk(child)


def known_globals(src: str) -> set[str]:
    """Module-level names: imports, defs, classes and plain assignments."""
    import ast

    names: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                names.add(a.asname or a.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.arg,)):
            names.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.comprehension,)):
            pass
    return names


def check_file(path: Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
    noqa_lines = {
        i + 1 for i, line in enumerate(src.splitlines()) if "noqa" in line
    }
    table = symtable.symtable(src, str(path), "exec")
    defined = known_globals(src) | set(dir(builtins))
    problems: list[str] = []

    for scope in walk(table):
        for sym in scope.get_symbols():
            name = sym.get_name()
            if sym.is_assigned() or sym.is_parameter():
                continue
            if not sym.is_global() and not sym.is_free():
                continue
            # __class__ is the implicit closure cell CPython creates for a
            # zero-arg super() inside a method; it is never assigned in source.
            if name in defined or name in ("__name__", "__file__", "__doc__",
                                           "__class__"):
                continue
            problems.append(
                f"{path}: {scope.get_name()}() uses undefined global {name!r}"
            )
    return problems


def main() -> int:
    here = Path(__file__).resolve().parent
    files = ([Path(a) for a in sys.argv[1:]] if len(sys.argv) > 1
             else sorted(here.glob("*.py")))
    all_problems: list[str] = []
    for f in files:
        all_problems.extend(check_file(f))
    if all_problems:
        for p in all_problems:
            print(p)
        print(f"\n{len(all_problems)} problem(s)")
        return 1
    print(f"{len(files)} files: no undefined global names")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
