#!/usr/bin/env python3
"""Keep secrets inside the newtypes that redact them.

Each secret is a newtype whose hand-written `Debug` and `Display` print
`<redacted>`, so any struct holding one can be logged with `{:?}` (see
docs/architecture.md, "Secrets"). The compiler does not defend that: a new
`api_key: String` field, or a `derive(Debug)` on a secret newtype, leaks. So this
check reads the declarations:

1. A struct or enum field whose name reads like a secret must be typed with a
   registered secret newtype (bare, or inside `Option`/`Vec`).
2. A registered secret newtype must hand-write `Debug` and `Display`, each
   rendering `<redacted>` and neither touching the raw value, and must not
   gain `Deref` (the escape hatch is `AsRef`).
3. A field excepted from rule 1 because it stages a value before validation
   must sit in a type that renders nothing.

A plural `tokens` is not a secret name: `parameter_max_tokens` counts LLM
tokens. Only type bodies are scanned, so function parameters and consts pass.
"""

import re
import subprocess
import sys
from pathlib import Path

# Adding a secret newtype means registering it here, which forces a review.
SECRET_TYPES = {
    "ClientPreauthKey": Path("crates/pipette-mgmt-client/src/types.rs"),
    "PrivateKey": Path("crates/pipette-mgmt-client/src/types.rs"),
    "UserPreauthKey": Path("crates/pipette-mgmt-client/src/types.rs"),
}

# Secret-sounding but not a credential ("token" also means a capability guard).
NON_SECRET_TYPES = {"TeardownToken"}

# Redacting newtypes from `pipette-plan-types` in pipette-mgmt. Rule 1
# accepts them; their shape is mgmt's to check.
FOREIGN_SECRET_TYPES = {"AuthToken"}

# Secret-named fields that stay a plain `String`, keyed `Type.field`: a staging
# struct decodes loosely so a bad key is named rather than reported as a
# malformed record. Sound only while the type renders nothing (rule 3).
UNVALIDATED_STAGING_FIELDS = {
    "BundleWire.private_key_hex": Path("crates/pipette-cli/src/identity/bundle.rs"),
}

# Expressions that reach the wrapped value; a rendering impl must not contain one.
RAW_ACCESS = ("as_ref", "self.0", "into_inner")

REDACTED = "<redacted>"

# `token` matches only in the singular (see the module docstring). `_hex` is how
# the bundle wire format spells the private key field.
SECRET_NAME = re.compile(
    r"^(?:[a-z0-9_]*_)?"
    r"(?:token|secret|password|passwd|credential|api_key|apikey|preauth_key"
    r"|private_key(?:_hex)?)$"
)

# `pub auth_token: Option<AuthToken>,` -> name, type. Struct-variant fields too.
FIELD = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?([a-z][a-z0-9_]*)\s*:\s*(.+?),?\s*$")

# Captures the type name so a field can be keyed `Type.field`.
TYPE_OPEN = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum)\s+([A-Za-z_][A-Za-z0-9_]*)"
)

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


def type_body_fragments(path: Path) -> list[tuple[int, str, str]]:
    """Field-declaration candidates inside a `struct`/`enum` body, with the name
    of the type declaring them.

    Brace counting, not parsing. A one-line body (`struct Foo { a: u8 }`) is
    split after its brace.
    """
    fragments: list[tuple[int, str, str]] = []
    depth = 0
    enclosing = ""
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if depth == 0:
            if not ((opened := TYPE_OPEN.match(line)) and "{" in line):
                continue
            enclosing = opened.group(1)
            inline = line.split("{", 1)[1].rstrip().removesuffix("}")
            fragments.extend((number, field, enclosing) for field in inline.split(","))
        else:
            fragments.append((number, line, enclosing))
        depth += line.count("{") - line.count("}")
    return fragments


def check_fields(files: list[Path]) -> list[str]:
    """Rule 1: a secret-named field is declared with a registered secret type."""
    violations = []
    secret = SECRET_TYPES.keys() | FOREIGN_SECRET_TYPES
    approved = secret | NON_SECRET_TYPES
    for path in files:
        for number, line, enclosing in type_body_fragments(path):
            if (stripped := line.strip()).startswith(("//", "#[")):
                continue
            if not (field := FIELD.match(line)):
                continue
            name, ty = field.group(1), field.group(2)
            if not SECRET_NAME.match(name) or unwrap(ty) in approved:
                continue
            if f"{enclosing}.{name}" in UNVALIDATED_STAGING_FIELDS:
                continue
            violations.append(
                f"{path}:{number}: field `{name}: {ty}` reads as a secret but is "
                f"not one of {', '.join(sorted(secret))}.\n"
                f"       Declare it with a redacting newtype, or, if it holds no "
                f"credential, add its type to NON_SECRET_TYPES in "
                f"{Path(__file__).name}.\n"
                f"       {stripped}"
            )
    return violations


def type_declaration(source: str, ty: str) -> re.Match[str] | None:
    """`struct BundleWire {`, the declaration rule 3 reads the derive list before."""
    return re.search(rf"^(?:pub(?:\([^)]*\))?\s+)?struct {ty}\b", source, re.M)


def rendering_impl(source: str, trait: str, ty: str) -> str | None:
    """The code of `impl std::fmt::<trait> for <ty>`, or `None` if absent.

    Bounded at the impl's closing brace so a later impl's `<redacted>` cannot
    vouch for this one, and stripped of `//` comments (also inside strings,
    which no redaction marker contains).
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

        # The `nutype` attribute preceding the declaration.
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


def check_staging_shape() -> list[str]:
    """Rule 3: a type excepted from rule 1 renders nothing.

    A `Debug` or `Display` on it, derived or hand-written, would print the
    raw secret the exception lets it hold.
    """
    violations = []
    for qualified, path in UNVALIDATED_STAGING_FIELDS.items():
        ty, field = qualified.split(".", 1)
        source = path.read_text() if path.exists() else ""
        if not (declaration := type_declaration(source, ty)):
            violations.append(
                f"{path}: expected to define `{ty}`; update "
                f"UNVALIDATED_STAGING_FIELDS in {Path(__file__).name} if it moved."
            )
            continue
        head = source[: declaration.start()]
        derives = head.rsplit("#[derive(", 1)[-1] if "#[derive(" in head else ""
        for trait in ("Debug", "Display"):
            if re.search(rf"\b{trait}\b", derives) or rendering_impl(source, trait, ty):
                violations.append(
                    f"{path}: `{ty}` renders {trait}, and `{ty}.{field}` holds a "
                    f"raw secret by exception.\n"
                    f"       Drop the {trait}, or type the field and remove the "
                    f"exception from {Path(__file__).name}."
                )
    return violations


def main() -> int:
    violations = (
        check_fields(tracked_rust_files())
        + check_newtype_shape()
        + check_staging_shape()
    )
    if not violations:
        return 0
    print("secret values must live in a redacting newtype:\n")
    for violation in violations:
        print(f"  {violation}")
    print('\nRule: docs/architecture.md, "Secrets".')
    return 1


if __name__ == "__main__":
    sys.exit(main())
