#!/usr/bin/env python3
"""Reject em dashes in user-visible client copy: Kotlin, Swift, and Rust.

An em dash is uncommon to type by hand, so it reads as machine-written in product
copy. This check covers the three client source trees; Markdown and Python are
each guarded separately.

One scanner for three languages, rather than a guard test per language. The three
started as separate implementations and immediately drifted: block comments nest
in all three, and two of the three tracked them with a boolean, so the inner `*/`
of `/* a /* b */ still */` ended the comment early and the tail was scanned as
copy. A single implementation cannot drift from itself. Running here also means
the rule is checked with no toolchain and no path filter, so a Swift-only change
no longer needs the expensive iOS job to have its copy checked.

Exempt:

- **Comments**, line and block, block comments nesting.
- **Logging.** `log::` in Rust, `Log.` in Kotlin, `AppLog.` in Swift: developer
  text that never reaches a screen. `HeadlessRunner.log` is deliberately NOT
  exempt, because it prints to a terminal a person reads.
- **Rust `#[cfg(test)]` modules.** Assertion messages are developer-facing, and
  fixture strings can be load-bearing: `pipette-doomloop` feeds an
  em-dash-prefixed body to a tail-drift detector as synthetic model output, so
  rewriting it would change what the detector is tested against.
- **A string literal that is exactly one em dash**, the empty-value glyph the
  mobile clients use in Settings rows and the Jobs results grid.

Two details are load-bearing and were paid for in real bugs:

1. A log call's arguments are blanked by CHARACTER, not by skipping the line. A
   line-level exemption lets `log::warn!("x"); return Err("a - b".into());`
   through, and that line's paren delta is zero, so no amount of depth
   bookkeeping catches it either.
2. String literals are tracked while scanning, char literals included. A `'"'` in
   `pipette-device/src/probe.rs` flipped string state on and never off, after
   which every later comment in the file was scanned as copy.

Fix a failure by rewriting the sentence, not by swapping in an en dash or a double
hyphen: those are the same tell wearing a hat. A period, comma, colon, or
parentheses almost always carries the same clause break.
"""

import re
import subprocess
import sys
from pathlib import Path

EM_DASH = chr(0x2014)

RUST_LOG_MACROS = (
    "log::trace!",
    "log::debug!",
    "log::info!",
    "log::warn!",
    "log::error!",
)
KOTLIN_LOG_PREFIXES = ("Log.d", "Log.w", "Log.e", "Log.i", "Log.v")


def rust_log_call(line: str, i: int) -> int | None:
    """Width of a `log::level!(` prefix at `i`, or None."""
    for macro in RUST_LOG_MACROS:
        if (
            line.startswith(macro, i)
            and line[i + len(macro) : i + len(macro) + 1] == "("
        ):
            return len(macro) + 1
    return None


def kotlin_log_call(line: str, i: int) -> int | None:
    """Width of a `Log.x(` prefix at `i`, or None. Whitespace before `(` is allowed."""
    for prefix in KOTLIN_LOG_PREFIXES:
        if not line.startswith(prefix, i):
            continue
        after = i + len(prefix)
        gap = len(line[after:]) - len(line[after:].lstrip())
        if line[after + gap : after + gap + 1] == "(":
            return len(prefix) + gap + 1
    return None


def swift_log_call(line: str, i: int) -> int | None:
    """Width of an `AppLog.<category>.<level>(` prefix at `i`, or None.

    Category and level are matched as any identifier, so a new one is covered the
    day it is added rather than the day someone remembers to list it.
    """
    marker = "AppLog."
    if not line.startswith(marker, i):
        return None
    at = i + len(marker)

    def identifier(pos: int) -> int:
        start = pos
        while pos < len(line) and (line[pos].isalnum() or line[pos] == "_"):
            pos += 1
        return pos if pos > start else -1

    at = identifier(at)
    if at < 0 or line[at : at + 1] != ".":
        return None
    at = identifier(at + 1)
    if at < 0:
        return None
    while at < len(line) and line[at] == " ":
        at += 1
    if line[at : at + 1] != "(":
        return None
    return at + 1 - i


def rust_string_open(line: str, i: int) -> tuple[int, str, bool] | None:
    """A Rust string opening at `i`, as (width, closing delimiter, escapes)."""
    if line[i] == "r" and not (i and (line[i - 1].isalnum() or line[i - 1] == "_")):
        hashes = 0
        while line[i + 1 + hashes : i + 2 + hashes] == "#":
            hashes += 1
        if line[i + 1 + hashes : i + 2 + hashes] == '"':
            # Raw strings carry their hash count because their contents routinely
            # include bare quotes.
            return hashes + 2, '"' + "#" * hashes, False
    if line[i] == '"':
        return 1, '"', True
    return None


def kotlin_string_open(line: str, i: int) -> tuple[int, str, bool] | None:
    if line.startswith('"""', i):
        return 3, '"""', False
    if line[i] == '"':
        return 1, '"', True
    return None


def swift_string_open(line: str, i: int) -> tuple[int, str, bool] | None:
    if line[i] == "#":
        hashes = 0
        while line[i + hashes : i + hashes + 1] == "#":
            hashes += 1
        if line[i + hashes : i + hashes + 1] == '"':
            multi = line.startswith('"""', i + hashes)
            width = hashes + (3 if multi else 1)
            return width, ('"""' if multi else '"') + "#" * hashes, False
        return None
    if line.startswith('"""', i):
        # A multi-line literal still processes escapes; only the `#` forms are raw.
        return 3, '"""', True
    if line[i] == '"':
        return 1, '"', True
    return None


def char_literal_width(line: str, i: int) -> int | None:
    """Width of a Rust/Kotlin char literal at `i`, or None.

    A char literal such as `'"'` or `'('` must not shift string state or paren
    depth. A Rust lifetime (`'a`) has no closing quote, so only skip when the
    shape really is a one-or-two-character literal.
    """
    if line[i] != "'":
        return None
    if line[i + 2 : i + 3] == "'":
        return 3
    if line[i + 1 : i + 2] == "\\" and line[i + 3 : i + 4] == "'":
        return 4
    return None


class Lang:
    """Everything that differs between the three clients."""

    def __init__(
        self,
        name,
        pattern,
        log_call,
        string_open,
        min_files,
        char_literals=False,
        cfg_test=False,
        padded_escape=True,
    ):
        self.name = name
        # Matched against tracked paths. Only the SHIPPING sources: a language's
        # own test tree is developer-facing, and excluding it is also what keeps
        # each guard's former test file out of scope now that they are gone.
        self.pattern = re.compile(pattern)
        self.log_call = log_call
        self.string_open = string_open
        self.min_files = min_files
        self.char_literals = char_literals
        self.cfg_test = cfg_test
        # Rust `\u{...}` and Swift `\u{...}` accept leading zeros, so a padded
        # spelling renders identically and must be caught. Kotlin's `\u` takes
        # EXACTLY four hex digits, so a padded form lexes as a different escape
        # followed by stray digits: not an em dash, and flagging it would be a
        # false positive.
        self.padded_escape = padded_escape


LANGS = [
    Lang(
        "rust",
        r"^crates/[^/]+/src/.*\.rs$",
        rust_log_call,
        rust_string_open,
        min_files=200,
        char_literals=True,
        cfg_test=True,
    ),
    Lang(
        "kotlin",
        r"^android/Pipette/app/src/main/java/.*\.kt$",
        kotlin_log_call,
        kotlin_string_open,
        min_files=50,
        char_literals=True,
        padded_escape=False,
    ),
    Lang(
        "swift",
        r"^ios/Pipette/Pipette/.*\.swift$",
        swift_log_call,
        swift_string_open,
        min_files=100,
    ),
]


def keep(code: list[str], ch: str, blank: bool) -> None:
    """Append `ch`, or a space when inside a log call's argument span.

    Module level, and told explicitly whether to blank, rather than a closure over
    the scanner's state: a closure would capture the loop variables late, which is
    the same class of bug the scanner exists to avoid.
    """
    code.append(" " if blank else ch)


def scan_lines(source: str, lang: Lang) -> list[tuple[str, bool]]:
    """Blank every span that is not user-visible copy.

    Returns one `(code, in_test)` per line. What survives in `code` is real code
    and the string literals a person can actually read.

    Kept as one function on purpose: it is a character-level state machine, and
    splitting it would spread the string, comment, brace-depth, and log-depth
    state across call boundaries, making exactly the desync bugs it exists to
    prevent easier to reintroduce.
    """
    out: list[tuple[str, bool]] = []
    # All three languages NEST block comments, so this counts depth rather than
    # holding a flag. With a flag, the inner `*/` of `/* a /* b */ still */` ends
    # the comment early and the tail gets scanned as copy.
    block_depth = 0
    closer: str | None = None  # Non-None while inside a string literal.
    escapes = False
    log_depth: int | None = None
    brace_depth = 0
    pending_test = False
    test_exit_depth: int | None = None

    for line in source.split("\n"):
        code: list[str] = []
        i = 0
        line_starts_in_test = test_exit_depth is not None

        while i < len(line):
            c = line[i]
            blank = log_depth is not None

            if block_depth > 0:
                if line.startswith("/*", i):
                    block_depth += 1
                    i += 2
                elif line.startswith("*/", i):
                    block_depth -= 1
                    i += 2
                else:
                    i += 1
                continue

            if closer is not None:
                if escapes and c == "\\":
                    keep(code, c, blank)
                    if i + 1 < len(line):
                        keep(code, line[i + 1], blank)
                    i += 2
                    continue
                if line.startswith(closer, i):
                    for k in range(len(closer)):
                        keep(code, line[i + k], blank)
                    i += len(closer)
                    closer = None
                    continue
                keep(code, c, blank)
                i += 1
                continue

            if line.startswith("/*", i):
                block_depth = 1
                i += 2
                continue
            if line.startswith("//", i):
                break

            if lang.cfg_test and c == "#" and line.startswith("#[cfg(test)]", i):
                # Arms the next `{` as the start of a test block.
                pending_test = True

            if log_depth is None:
                width = lang.log_call(line, i)
                if width is not None:
                    code.append(" " * width)
                    log_depth = 1
                    i += width
                    continue

            if lang.char_literals:
                width = char_literal_width(line, i)
                if width is not None:
                    for k in range(width):
                        keep(code, line[i + k], blank)
                    i += width
                    continue

            opened = lang.string_open(line, i)
            if opened is not None:
                width, closer, escapes = opened
                for k in range(width):
                    keep(code, line[i + k], blank)
                i += width
                continue

            if lang.cfg_test:
                # Brace bookkeeping in code context only, so braces inside strings
                # and comments cannot shift it.
                if c == "{":
                    if pending_test:
                        pending_test = False
                        test_exit_depth = brace_depth
                    brace_depth += 1
                elif c == "}":
                    brace_depth -= 1
                    if test_exit_depth == brace_depth:
                        test_exit_depth = None

            if log_depth is not None:
                if c == "(":
                    log_depth += 1
                elif c == ")":
                    log_depth -= 1
                    if log_depth <= 0:
                        log_depth = None
                code.append(" ")
                i += 1
                continue

            code.append(c)
            i += 1

        # A line is test code if it was inside the block on entry or opened it, so
        # the `#[cfg(test)]` and `mod tests {` lines are covered too.
        in_test = line_starts_in_test or test_exit_depth is not None or pending_test
        out.append(("".join(code), in_test))
    return out


def contains_padded(haystack: str, prefix: str, digits: str) -> bool:
    """True when `prefix` appears followed by optional zeros and then `digits`."""
    at = haystack.find(prefix)
    while at != -1:
        if haystack[at + len(prefix) :].lstrip("0").startswith(digits):
            return True
        at = haystack.find(prefix, at + 1)
    return False


def has_em_dash(line: str, lang: Lang) -> bool:
    """Every rendering this guard rejects.

    The literal character is not enough: an escape or an HTML entity renders as one
    too. The needles are built from the code point so this file does not contain
    its own.
    """
    # A lone em dash is the empty-value placeholder, not punctuation.
    stripped = line.replace(f'"{EM_DASH}"', '""')
    if EM_DASH in stripped:
        return True
    lower = stripped.lower()
    if lang.padded_escape:
        if contains_padded(lower, "\\u{", f"{0x2014:x}}}"):
            return True
    elif "\\u" + f"{0x2014:04x}" in lower:
        return True
    # Lowercased because `&MDASH;` renders the same as `&mdash;`.
    if "&" + "mdash" + ";" in lower:
        return True
    # HTML numeric references DO allow leading zeros, and a padded form renders
    # exactly like its unpadded twin, so skip any padding run before the digits.
    return any(
        contains_padded(lower, prefix, digits)
        for prefix, digits in (("&#", f"{8212};"), ("&#x", f"{0x2014:x};"))
    )


def flagged(source: str, lang: Lang) -> list[str]:
    """Run the real detector over an in-memory source and return what it flags."""
    return [
        code.strip()
        for code, in_test in scan_lines(source, lang)
        if not in_test and has_em_dash(code, lang)
    ]


def tracked() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout.split("\n")
    return [p for p in out if p]


def sources(lang: Lang, paths: list[str]) -> list[Path]:
    return [Path(p) for p in paths if lang.pattern.match(p)]


def offenders(path: Path, lang: Lang) -> list[str]:
    text = path.read_text(errors="replace")
    return [
        f"{path}:{n}: {code.strip()[:110]}"
        for n, (code, in_test) in enumerate(scan_lines(text, lang), 1)
        if not in_test and has_em_dash(code, lang)
    ]


def selftest() -> list[str]:
    """Fixtures for the exemptions, run on every invocation.

    An exemption that is too eager still exits 0, so nothing surfaces it. These
    cases are the ones that have actually been wrong at some point.
    """
    d = EM_DASH
    rust, kotlin, swift = LANGS
    # Composed rather than written out, for the same reason the needles are: a
    # fixture holding the entity literally would make this file trip its own check.
    ent = "&" + "mdash" + ";"
    ent_upper = "&" + "MDASH" + ";"
    num_padded = "&#" + f"0{8212};"
    hex_ent = "&#x" + f"{0x2014:x};"
    brace_escape = "\\u{" + f"{0x2014:x}" + "}"
    brace_escape_padded = "\\u{" + f"000{0x2014:x}" + "}"
    kotlin_escape = "\\u" + f"{0x2014:04x}"
    kotlin_escape_padded = "\\u" + f"00{0x2014:04x}"

    cases: list[tuple[Lang, str, int, str]] = [
        # --- shared shapes, checked per language -------------------------------
        (rust, f'let m = "port in use {d} try again";', 1, "rust: plain offender"),
        (kotlin, f'val m = "port in use {d} try again"', 1, "kotlin: plain offender"),
        (swift, f'let m = "port in use {d} try again"', 1, "swift: plain offender"),
        (rust, f"// a comment {d} here\nlet x = 1;", 0, "rust: line comment"),
        (kotlin, f"// a comment {d} here\nval x = 1", 0, "kotlin: line comment"),
        (swift, f"/// a doc comment {d} here\nlet x = 1", 0, "swift: doc comment"),
        (rust, f"/* spans\n * a {d} break\n */\nlet x = 1;", 0, "rust: block comment"),
        (
            kotlin,
            f"/* spans\n * a {d} break\n */\nval x = 1",
            0,
            "kotlin: block comment",
        ),
        (swift, f"/* spans\n * a {d} break\n */\nlet x = 1", 0, "swift: block comment"),
        # --- nested block comments: all three languages nest -------------------
        (
            rust,
            f"/* outer /* inner */ still a {d} comment */\nlet x = 1;",
            0,
            "rust: NESTED block comment stays exempt",
        ),
        (
            kotlin,
            f"/* outer /* inner */ still a {d} comment */\nval x = 1",
            0,
            "kotlin: NESTED block comment stays exempt",
        ),
        (
            swift,
            f"/* outer /* inner */ still a {d} comment */\nlet x = 1",
            0,
            "swift: NESTED block comment stays exempt",
        ),
        (
            rust,
            f'/* outer /* inner */ */\nlet s = "a {d} b";',
            1,
            "rust: code AFTER a nested comment is still checked",
        ),
        (
            swift,
            f'/* outer /* inner */ */\nlet s = "a {d} b"',
            1,
            "swift: code AFTER a nested comment is still checked",
        ),
        # --- logging is developer text -----------------------------------------
        (rust, f'log::warn!("retrying {d} backoff");', 0, "rust: log macro"),
        (kotlin, f'Log.w(TAG, "retrying {d} backoff")', 0, "kotlin: Log call"),
        (swift, f'AppLog.jobRun.info("retrying {d} backoff")', 0, "swift: AppLog"),
        (
            swift,
            f'AppLog.storage.info(\n    "wrapped {d} message"\n)',
            0,
            "swift: multi-line AppLog call",
        ),
        (
            rust,
            f'log::warn!("x"); return Err("a {d} b".into());',
            1,
            "rust: a log call must not exempt the REST of its line",
        ),
        (
            swift,
            f'AppLog.jobRun.info("x"); throw fail("a {d} b")',
            1,
            "swift: a log call must not exempt the REST of its line",
        ),
        (
            swift,
            f'HeadlessRunner.log("a {d} b")',
            1,
            "swift: headless output is a terminal a person reads, NOT exempt",
        ),
        (rust, f'println!("a {d} b");', 1, "rust: println is user-visible"),
        # --- the empty-value glyph ---------------------------------------------
        (kotlin, f'PropertyRow("Email", value ?: "{d}")', 0, "kotlin: lone glyph"),
        (swift, f'Text(value ?? "{d}")', 0, "swift: lone glyph"),
        (
            kotlin,
            f'val a = "{d}"; val b = "real {d} prose"',
            1,
            "kotlin: the glyph does not excuse prose on the same line",
        ),
        # --- strings that must not desync the scan -----------------------------
        (
            rust,
            f'let q = \x27"\x27;\n// a comment {d} here\n',
            0,
            "rust: a CHAR LITERAL holding a quote must not flip string state",
        ),
        (
            kotlin,
            f'val q = \x27"\x27\n// a comment {d} here\n',
            0,
            "kotlin: a char literal holding a quote must not flip string state",
        ),
        (
            rust,
            f'let s = "// not a comment {d}";',
            1,
            "rust: a // inside a string does not start a comment",
        ),
        (rust, f'let s = r#"raw {d} here"#;', 1, "rust: raw string"),
        (kotlin, f'val s = """raw {d} here"""', 1, "kotlin: triple-quoted string"),
        (swift, f'let s = #"raw {d} here"#', 1, "swift: raw string"),
        (swift, f'let s = """\nmulti {d} line\n"""', 1, "swift: multi-line string"),
        (
            rust,
            f'let s = r#"has " quote {d}"#;',
            1,
            "rust: a bare quote inside a raw string does not close it",
        ),
        # --- Rust test code is exempt ------------------------------------------
        (
            rust,
            f'#[cfg(test)]\nmod tests {{\n    let body = "a {d} b";\n}}\n',
            0,
            "rust: cfg(test) fixtures can be load-bearing",
        ),
        (
            rust,
            '#[cfg(test)]\nmod tests {\n    let s = "{}";\n}\n'
            f'let real = "a {d} b";\n',
            1,
            "rust: a brace in a string must not end the test module early",
        ),
        # --- every rendering ----------------------------------------------------
        (rust, f'let s = "a {ent} b";', 1, "rust: html entity"),
        (kotlin, f'val s = "a {ent_upper} b"', 1, "kotlin: entity, uppercase"),
        (swift, f'let s = "a {num_padded} b"', 1, "swift: zero-padded entity"),
        (rust, f'let s = "a {hex_ent} b";', 1, "rust: hex entity"),
        (rust, f'let s = "a {brace_escape} b";', 1, "rust: brace escape"),
        (
            rust,
            f'let s = "a {brace_escape_padded} b";',
            1,
            "rust: brace escape accepts leading zeros",
        ),
        (swift, f'let s = "a {brace_escape} b"', 1, "swift: brace escape"),
        (kotlin, f'val s = "a {kotlin_escape} b"', 1, "kotlin: four-digit escape"),
        (
            kotlin,
            f'val s = "a {kotlin_escape_padded} b"',
            0,
            "kotlin: a PADDED escape is a different escape, not an em dash",
        ),
        # --- clean sources ------------------------------------------------------
        (rust, 'let s = "plain";', 0, "rust: no glyph"),
        (kotlin, 'val s = "plain"', 0, "kotlin: no glyph"),
        (swift, 'let s = "plain"', 0, "swift: no glyph"),
    ]

    failures = []
    for lang, source, want, why in cases:
        got = len(flagged(source, lang))
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

    paths = tracked()
    hits: list[str] = []
    for lang in LANGS:
        files = sources(lang, paths)
        if len(files) < lang.min_files:
            # A broken walk would make the check vacuously green.
            print(f"expected to walk the {lang.name} sources, found {len(files)} files")
            return 1
        hits.extend(h for f in files for h in offenders(f, lang))

    if not hits:
        return 0
    print(f"em dashes in user-visible client copy ({len(hits)}):")
    for h in hits:
        print(f"  {h}")
    print()
    print("Rewrite the sentence instead of swapping in an en dash or '--'.")
    print("Comments, logging, Rust #[cfg(test)] code, and a string that is exactly")
    print("one em dash are exempt.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
