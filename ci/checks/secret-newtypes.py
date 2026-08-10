#!/usr/bin/env python3
"""Keep secrets inside the newtypes that redact them.

Two secret values cross this workspace — a HuggingFace access token and a
pre-auth registration key — and each is a newtype whose `Debug` and `Display`
are hand-written to print `<redacted>` (see `docs/architecture.md`, "Secrets").
That is what makes it safe for the client to log a whole `RunRequest` with
`{:?}`: the leaf hides itself, so every struct holding one inherits the
redaction.

Nothing in the compiler defends that arrangement. A field added as
`api_key: String` gets a derived `Debug` and is published by the first log line
that dumps its parent, and a `derive(Debug)` slipped into a secret newtype's
`nutype` list silently replaces the hand-written redaction. A unit test can't
see either one: it can only assert about a token it plants itself, in a field
that already exists.

So this check reads the declarations instead:

1. A struct or enum field whose name reads like a secret must be typed with a
   registered secret newtype (bare, or inside `Option`/`Vec`).
2. A registered secret newtype must hand-write `Debug` and `Display`, each
   rendering `<redacted>` and neither touching the raw value, and must not
   gain `Deref` — the escape hatch is `AsRef`.

Field-name matching deliberately treats a plural `tokens` as *not* a secret:
`parameter_max_tokens` and friends count LLM tokens, and they outnumber the
real secrets in this tree. Only scans type bodies, so a `token: &str` function
parameter or a `const HF_TOKEN_ENV` naming an env var is not a violation.
"""

import re
import subprocess
import sys
from pathlib import Path

# Registered secret newtypes: the type a secret-named field must be declared
# with, and whose own shape rule 2 enforces. Adding a secret means adding it
# here — which is the review moment this check exists to force.
SECRET_TYPES = {
    "AuthToken": Path("crates/pipette-plan-types/src/primitives.rs"),
    "PreauthKey": Path("crates/pipette-mgmt-client/src/types.rs"),
}

# Types that answer a secret-sounding field name but hold no credential —
# "token" also means a capability guard. Reviewed exceptions, so a secret-named
# field typed `String` still fails.
NON_SECRET_TYPES = {"TeardownToken"}

# Expressions that reach the wrapped value. A rendering impl containing one is
# printing the secret, whatever else it also prints.
RAW_ACCESS = ("as_ref", "self.0", "into_inner")

REDACTED = "<redacted>"

# A field name that reads like it holds a credential. `token` matches only in
# the singular — see the module docstring.
SECRET_NAME = re.compile(
    r"^(?:[a-z0-9_]*_)?"
    r"(?:token|secret|password|passwd|credential|api_key|apikey|preauth_key)$"
)

# `pub auth_token: Option<AuthToken>,` → name, type. Struct fields and
# struct-variant fields read the same.
FIELD = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?([a-z][a-z0-9_]*)\s*:\s*(.+?),?\s*$")

# `struct Foo {` / `enum Bar {` — the bodies rule 1 scans.
TYPE_OPEN = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum)\s+[A-Za-z_][A-Za-z0-9_]*"
)

# The type a field must resolve to once wrappers are peeled off.
WRAPPER = re.compile(r"^(?:Option|Vec)<(.+)>$")


def tracked_rust_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "crates/*.rs"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [Path(f) for f in out]


def unwrap(ty: str) -> str:
    """`Option<Vec<AuthToken>>` → `AuthToken`; leaves anything else alone."""
    while match := WRAPPER.match(ty.strip()):
        ty = match.group(1)
    return ty.strip()


def type_body_fragments(path: Path) -> list[tuple[int, str]]:
    """Field-declaration candidates inside a `struct`/`enum` body.

    Brace counting, not parsing: it only has to answer "could this be a field
    declaration in a type body", which excludes function parameters and consts.
    A body opening and closing on one line (`struct Foo { a: u8 }`) still
    declares fields, so the text after the brace is split and kept.
    """
    fragments: list[tuple[int, str]] = []
    depth = 0
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if depth == 0:
            if not (TYPE_OPEN.match(line) and "{" in line):
                continue
            inline = line.split("{", 1)[1].rstrip().removesuffix("}")
            fragments.extend((number, field) for field in inline.split(","))
        else:
            fragments.append((number, line))
        depth += line.count("{") - line.count("}")
    return fragments


def check_fields(files: list[Path]) -> list[str]:
    """Rule 1: a secret-named field is declared with a registered secret type."""
    violations = []
    approved = SECRET_TYPES.keys() | NON_SECRET_TYPES
    for path in files:
        for number, line in type_body_fragments(path):
            if (stripped := line.strip()).startswith(("//", "#[")):
                continue
            if not (field := FIELD.match(line)):
                continue
            name, ty = field.group(1), field.group(2)
            if not SECRET_NAME.match(name) or unwrap(ty) in approved:
                continue
            violations.append(
                f"{path}:{number}: field `{name}: {ty}` reads as a secret but is "
                f"not one of {', '.join(sorted(SECRET_TYPES))}.\n"
                f"       Declare it with a redacting newtype, or, if it holds no "
                f"credential, add its type to NON_SECRET_TYPES in "
                f"{Path(__file__).name}.\n"
                f"       {stripped}"
            )
    return violations


def rendering_impl(source: str, trait: str, ty: str) -> str | None:
    """The code of `impl std::fmt::<trait> for <ty>`, or `None` if absent.

    Bounded at the impl's closing brace so a `<redacted>` belonging to some
    later impl can't vouch for this one, and stripped of comments so prose
    about `as_ref` neither implicates nor absolves the impl. Stripping is
    line-based, which would also cut a `//` inside a string literal — no
    redaction marker contains one.
    """
    start = source.find(f"impl std::fmt::{trait} for {ty} {{")
    if start == -1:
        return None
    end = source.find("\n}", start)
    code = source[start:] if end == -1 else source[start:end]
    return "\n".join(line.split("//", 1)[0] for line in code.splitlines())


def check_newtype_shape() -> list[str]:
    """Rule 2: the secret newtypes keep hand-written, redacting renderings."""
    violations = []
    for ty, path in SECRET_TYPES.items():
        if not path.exists():
            violations.append(
                f"{path}: expected to define `{ty}`; update SECRET_TYPES in "
                f"{Path(__file__).name} if it moved."
            )
            continue
        source = path.read_text()

        # The `nutype`/`derive` list preceding the declaration. Rendering has to
        # be hand-written, so a derived one is the bug.
        head = source.split(f"struct {ty}(")[0].rsplit("#[nutype", 1)
        derives = head[-1] if len(head) > 1 else ""

        for trait in ("Debug", "Display"):
            if re.search(rf"\b{trait}\b", derives):
                violations.append(
                    f"{path}: `{ty}` derives {trait}, which renders the secret.\n"
                    f"       Drop it from the derive list and hand-write "
                    f"`impl std::fmt::{trait} for {ty}` printing `{REDACTED}`."
                )
                continue
            body = rendering_impl(source, trait, ty)
            if body is None:
                violations.append(
                    f"{path}: `{ty}` has no hand-written {trait} printing "
                    f"`{REDACTED}`.\n"
                    f"       Without it a dump, or a `{{}}`, publishes the secret."
                )
                continue
            if REDACTED not in body:
                violations.append(
                    f"{path}: `{ty}`'s {trait} does not print `{REDACTED}`.\n"
                    f"       Whatever it renders instead reaches the logs."
                )
            if leak := next((r for r in RAW_ACCESS if r in body), None):
                violations.append(
                    f"{path}: `{ty}`'s {trait} reaches the raw value via "
                    f"`{leak}`.\n"
                    f"       A rendering impl must not touch it; `AsRef` is for "
                    f"callers that need the real token."
                )

        if re.search(rf"impl std::ops::Deref for {ty}\b", source) or re.search(
            r"\bDeref\b", derives
        ):
            violations.append(
                f"{path}: `{ty}` exposes `Deref`, which re-exposes the raw value.\n"
                f"       `AsRef` is the sanctioned door; see docs/architecture.md "
                f'("Secrets").'
            )
    return violations


def main() -> int:
    violations = check_fields(tracked_rust_files()) + check_newtype_shape()
    if not violations:
        return 0
    print("secret values must live in a redacting newtype:\n")
    for violation in violations:
        print(f"  {violation}")
    print('\nRule: docs/architecture.md, "Secrets".')
    return 1


if __name__ == "__main__":
    sys.exit(main())
