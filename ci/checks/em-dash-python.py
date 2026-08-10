#!/usr/bin/env python3
"""Reject em dashes in Python string literals a person reads.

Companion to `em-dash-docs.py` (Markdown) and `em-dash-clients.py` (Kotlin, Swift,
Rust). Python in this repo is tooling rather than product, but its `print` output
and error messages are still text a developer reads, and the house style covers
them.

Unlike the client guard, this one needs no hand-rolled scanner: `tokenize` gives
the exact token stream, so comments never reach the check and a quote inside a
string can never desync it. That is the same failure `em-dash-clients.py` has to
defend against by hand, where a `'"'` char literal in Rust flips string state on
and never off.

Exempt:

- **Comments.** `tokenize` reports them as their own token type.
- **Docstrings**, meaning a bare string expression opening a module, class, or
  function. Prose for the reader of the source, like a comment.
- **A string that is exactly one em dash**, the empty-value glyph the clients use.

Every rendering is rejected, not just U+2014, and the needles are built from the
code point so this file does not trip its own check.

Fix by rewriting the sentence. A period, comma, colon, or parentheses almost
always carries the same clause break; an en dash or `--` is the same tell wearing
a hat.
"""

import io
import subprocess
import sys
import token as tokmod
import tokenize
from pathlib import Path

EM_DASH = chr(0x2014)

# A docstring can only be the FIRST statement of a module, class, or function, so
# just three predecessors are possible: `ENCODING` (module docstring, the file's
# first token), `INDENT` (the opening statement of a class or function body), and
# `None` (nothing seen yet).
#
# `NEWLINE` and `DEDENT` are deliberately absent even though a real docstring is
# never preceded by them. Including either exempts an ordinary string EXPRESSION
# statement that merely follows another statement, which is not a docstring and
# whose text a reader may well see. `NL` is absent for the same reason: inside a
# call, a wrapped string argument is preceded by `NL`, and counting that as a
# statement start exempted every wrapped message (it under-reported 12 real hits
# as 4).
DOCSTRING_PREDECESSORS = {
    None,
    tokmod.INDENT,
    tokmod.ENCODING,
}

OPENERS = {"(", "[", "{"}
CLOSERS = {")", "]", "}"}


def tracked_python() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "*.py"], capture_output=True, text=True, check=True
    ).stdout.split()
    # Vendored trees are not ours to style.
    return sorted(p for p in out if "/vendor/" not in p and not p.startswith("vendor/"))


def renderings() -> list[tuple[str, str]]:
    """Prefix/digits pairs so ZERO-PADDED spellings are caught too."""
    return [
        ("&", f"{'mdash'};"),
        ("&#", f"{8212};"),
        ("&#x", f"{0x2014:x};"),
        ("\\u", f"{0x2014:04x}"),
        ("\\N{EM DASH", "}"),
    ]


def contains_padded(haystack: str, prefix: str, digits: str) -> bool:
    at = haystack.find(prefix)
    while at != -1:
        if haystack[at + len(prefix) :].lstrip("0").startswith(digits):
            return True
        at = haystack.find(prefix, at + 1)
    return False


PREFIX_LETTERS = "rbufRBUF"
QUOTES = ('"""', "'''", '"', "'")


def literal_body(text: str) -> str:
    """The text inside a string literal's quotes.

    Stripping the quote and prefix characters as a SET is wrong: `str.strip` keeps
    eating, so a literal holding an em dash then a prefix letter collapses to the
    bare glyph and wins the exemption below. Take the prefix off the front only,
    then remove one matching quote run from each end.
    """
    i = 0
    while i < len(text) and text[i] in PREFIX_LETTERS:
        i += 1
    rest = text[i:]
    for quote in QUOTES:
        if (
            rest.startswith(quote)
            and rest.endswith(quote)
            and len(rest) >= 2 * len(quote)
        ):
            return rest[len(quote) : -len(quote)]
    return rest


def has_em_dash(text: str) -> bool:
    # A lone em dash is the empty-value glyph, not punctuation.
    if literal_body(text).strip() == EM_DASH:
        return False
    if EM_DASH in text:
        return True
    lower = text.lower()
    return any(contains_padded(lower, p.lower(), d.lower()) for p, d in renderings())


def offenders(path: str) -> list[str]:
    return scan_source(Path(path).read_text(errors="replace"), path)


def scan_source(src: str, path: str) -> list[str]:
    hits: list[str] = []
    previous = None
    depth = 0
    try:
        stream = tokenize.generate_tokens(io.StringIO(src).readline)
        for tok in stream:
            if tok.type in (tokmod.STRING, getattr(tokmod, "FSTRING_MIDDLE", -1)):
                is_docstring = depth == 0 and previous in DOCSTRING_PREDECESSORS
                if not is_docstring and has_em_dash(tok.string):
                    hits.append(f"{path}:{tok.start[0]}: {tok.string.strip()[:110]}")
            if tok.type == tokmod.OP:
                if tok.string in OPENERS:
                    depth += 1
                elif tok.string in CLOSERS:
                    depth = max(0, depth - 1)
            # `NL` and comments do not end a statement, so they must not become the
            # predecessor a docstring is judged against.
            if tok.type not in (tokmod.COMMENT, tokmod.NL):
                previous = tok.type
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        print(f"{path}: could not tokenize: {exc}")
        return [f"{path}: unparseable"]
    return hits


def selftest() -> list[str]:
    """Fixtures for the two exemptions, both of which have been wrong before.

    Runs on every invocation. It costs microseconds, and it is the only thing
    standing between a subtle scanner bug and a check that passes while missing
    real hits.
    """
    d = EM_DASH
    # Every spelling is COMPOSED rather than written out, for the same reason the
    # needles are: a fixture holding `&mdash;` literally would make this file trip
    # its own check. Splitting the text across concatenated tokens keeps any single
    # token clean while the assembled fixture source is what gets scanned.
    ent = "&" + "mdash" + ";"
    ent_upper = "&" + "MDASH" + ";"
    num = "&#" + f"{8212};"
    num_padded = "&#" + f"0{8212};"
    hex_ent = "&#x" + f"{0x2014:x};"
    escape = "\\" + "u" + f"{0x2014:04x}"
    named = "\\" + "N{" + "EM DASH" + "}"
    cases: list[tuple[str, int, str]] = [
        # (source, expected hit count, what it pins down)
        (f'x = "a {d} b"\n', 1, "a plain offender"),
        (f'x = "{d}"\n', 0, "a lone glyph is the empty-value placeholder"),
        (
            f'x = "{d}b"\n',
            1,
            "a glyph plus a PREFIX LETTER must not collapse to the placeholder",
        ),
        (f'x = "{d}buff"\n', 1, "several prefix letters, same trap"),
        (f'"""module {d} docstring"""\n', 0, "module docstring"),
        (f'def f():\n    """doc {d} here"""\n', 0, "function docstring"),
        (f'class C:\n    """doc {d} here"""\n', 0, "class docstring"),
        (f"# a comment {d} here\nx = 1\n", 0, "comment"),
        (
            f'raise SystemExit(\n    "wrapped {d} message"\n)\n',
            1,
            "a wrapped argument is preceded by NL but is NOT a docstring",
        ),
        (
            f'x = 1\n"bare {d} statement"\n',
            1,
            "a string statement after another statement is NOT a docstring",
        ),
        (
            f'def f():\n    """doc"""\n    "second {d} string"\n',
            1,
            "only the FIRST string in a body is a docstring",
        ),
        (f'x = "a {ent} b"\n', 1, "html entity"),
        (f'x = "a {ent_upper} b"\n', 1, "html entity, uppercase"),
        (f'x = "a {num} b"\n', 1, "numeric entity"),
        (f'x = "a {num_padded} b"\n', 1, "zero-padded numeric entity"),
        (f'x = "a {hex_ent} b"\n', 1, "hex entity"),
        (f'x = "a {escape} b"\n', 1, "unicode escape"),
        (f'x = "a {named} b"\n', 1, "named escape"),
        (f'x = f"a {d} b"\n', 1, "f-string"),
        (f'x = """a {d} b"""\n', 1, "triple-quoted assignment is not a docstring"),
        ('x = "plain"\n', 0, "no glyph"),
    ]
    failures = []
    for src, want, why in cases:
        got = len(scan_source(src, "<selftest>"))
        if got != want:
            failures.append(f"selftest: expected {want} hit(s), got {got} for {why}")
    return failures


def main() -> int:
    failures = selftest()
    if failures:
        print("the guard's own fixtures failed, so its verdict cannot be trusted:")
        for f in failures:
            print(f"  {f}")
        return 1
    files = tracked_python()
    if len(files) < 5:
        # A broken walk would make the check vacuously green.
        print(f"expected to scan the Python sources, found {len(files)} files")
        return 1
    hits = [h for f in files for h in offenders(f)]
    if not hits:
        return 0
    print(f"em dashes in Python strings a person reads ({len(hits)}):")
    for h in hits:
        print(f"  {h}")
    print()
    print("Rewrite the sentence instead of swapping in an en dash or '--'.")
    print("Comments, docstrings, and a string that is exactly one em dash are exempt.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
