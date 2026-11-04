#!/usr/bin/env python3
"""Enforce the Problem / Solution / Testing structure in a pull request body.

What passes:

1. `**Problem**`, `**Solution**` and `**Testing**` each appear exactly once, as
   a heading (a line that is only the bold span). `## Problem` is rejected so
   there is one accepted form, and bold is the one this repository writes:
   134 of 134 core-section headings across 45 merged PRs.
2. They appear in that order.
3. Each carries content before the next heading, not just the template prompt.
   A fenced code block counts.

Extra sections are free, in any form and position. Headings inside a fenced
code block are not headings. Section length, the `Closes PIP-123` trailer and
the PR title are not checked.

    ./ci/pr-body.py                  # the current branch's PR, via gh
    ./ci/pr-body.py --pr 1234        # a specific PR, via gh
    ./ci/pr-body.py body.md          # a file holding the body
    ./ci/pr-body.py -                # stdin
    ./ci/pr-body.py --env PR_BODY    # what CI uses

Not in `ci/checks/` because a PR description is not a tracked file. It runs
from .github/workflows/pr-body.yml so it re-runs when the description is edited.

Exit status: 0 clean, 1 on a violation, 2 when no body could be read.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

SECTIONS = ("Problem", "Solution", "Testing")

# Only the bold span on its line, so `**Problem**` opening a sentence stays
# prose. Up to three leading spaces, as in Markdown.
BOLD = re.compile(r"^ {0,3}(?:\*\*(.+?)\*\*|__(.+?)__)[ \t]*$")
ATX = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})")

# What a fenced line becomes. Not a blank, so a section holding only a code
# block is not reported empty.
FENCED = "\x00"

# The prompts .github/PULL_REQUEST_TEMPLATE.md ships; left alone they are not an answer.
PROMPTS = {
    "what problem does this solve?",
    "what changed?",
    "what did you run?",
}

SKELETON = """\
**Problem**

What was wrong.

**Solution**

What the change does about it.

**Testing**

What you ran."""


def normalize(text: str) -> str:
    """Fold a heading to its comparable form: lowercase, no trailing colon."""
    return re.sub(r"[ \t]+", " ", text).strip().rstrip(":.").strip().lower()


def strip_fences(body: str) -> list[str]:
    """Body lines with every fenced code block replaced by `FENCED`.

    As GitHub renders it: a fence closes only on a marker at least as long as
    its opener, and an unclosed fence runs to the end of the body.
    """
    out: list[str] = []
    open_marker: str | None = None
    for line in body.splitlines():
        m = FENCE.match(line)
        if open_marker is None:
            if m:
                open_marker = m.group(1)
                out.append(FENCED)
                continue
        else:
            closes = (
                m
                and m.group(1)[0] == open_marker[0]
                and len(m.group(1)) >= len(open_marker)
            )
            out.append(FENCED)
            if closes:
                open_marker = None
            continue
        out.append(line)
    return out


def headings(lines: list[str]) -> list[tuple[int, str, str, str]]:
    """Every heading, as (line index, style, normalized title, marker).

    The marker is what the author typed (`##`, `**`), so a message can quote it.
    """
    found = []
    for i, line in enumerate(lines):
        m = BOLD.match(line)
        if m:
            marker = "**" if m.group(1) else "__"
            found.append((i, "bold", normalize(m.group(1) or m.group(2)), marker))
            continue
        m = ATX.match(line)
        if m:
            found.append((i, "atx", normalize(m.group(2)), m.group(1)))
    return found


def content(lines: list[str], start: int, stop: int) -> list[str]:
    """The lines of a section: after its heading, up to the next one."""
    return [ln for ln in lines[start + 1 : stop] if ln.strip()]


def check(body: str) -> list[str]:
    """Every violation in `body`, each stating the rule and the fix."""
    if not body.strip():
        return ["The description is empty."]

    lines = strip_fences(body)
    found = headings(lines)
    bounds = [h[0] for h in found] + [len(lines)]

    bold: dict[str, list[int]] = {}
    atx: dict[str, str] = {}
    for n, (_, style, title, marker) in enumerate(found):
        for section in SECTIONS:
            if title != section.lower():
                continue
            if style == "bold":
                bold.setdefault(section, []).append(n)
            else:
                atx.setdefault(section, marker)

    problems = []
    for section in SECTIONS:
        occurrences = bold.get(section, [])
        marker = atx.get(section)
        if marker and not occurrences:
            problems.append(
                f"{section}: written as `{marker} {section}`. "
                f"Section headings here are bold: `**{section}**`."
            )
        elif marker:
            problems.append(
                f"{section}: `{marker} {section}` sits beside the `**{section}**` "
                "heading. Drop it; one heading per section."
            )
        if not occurrences:
            if not marker:
                problems.append(f"{section}: no `**{section}**` heading.")
            continue
        if len(occurrences) > 1:
            problems.append(
                f"{section}: {len(occurrences)} `**{section}**` headings. "
                "Exactly one of each, so the section a reader lands on is the section."
            )
        n = occurrences[0]
        body_lines = content(lines, found[n][0], bounds[n + 1])
        if not body_lines:
            problems.append(f"{section}: the heading is there, the section is empty.")
        elif all(normalize(ln) in PROMPTS for ln in body_lines):
            prompt = body_lines[0].strip()
            problems.append(
                f"{section}: still holds the template prompt ({prompt!r}). "
                "Replace it with the answer."
            )

    order = [bold[s][0] for s in SECTIONS if len(bold.get(s, [])) == 1]
    if len(order) == len(SECTIONS) and order != sorted(order):
        got = sorted(SECTIONS, key=lambda s: bold[s][0])
        problems.append(
            f"Out of order: {', '.join(got)}. "
            f"Expected {', '.join(SECTIONS)}, so every description reads the same way."
        )

    return problems


def selftest() -> list[str]:
    """Fixtures for the parser, so a broken one cannot pass every body.

    Each case is (body, expected number of violations).
    """
    good = "**Problem**\np\n\n**Solution**\ns\n\n**Testing**\nt\n"
    cases = [
        (good, 0),
        # Extra sections, before and between, in either form.
        ("Stacked on #1.\n\n" + good + "\n## Notes\nn\n", 0),
        (
            "**Problem**\np\n\n**Provenance**\nx\n\n"
            "**Solution**\ns\n\n**Testing**\nt\n",
            0,
        ),
        # __bold__ is the same heading.
        ("__Problem__\np\n\n__Solution__\ns\n\n__Testing__\nt\n", 0),
        # A list, a table or a quote is content.
        ("**Problem**\n- p\n\n**Solution**\n| a |\n\n**Testing**\n> t\n", 0),
        # So is a code block on its own: the commands you ran are the answer.
        ("**Problem**\np\n\n**Solution**\ns\n\n**Testing**\n```\nmake\n```\n", 0),
        # A prompt the author answered under rather than deleting.
        (
            "**Problem**\nWhat problem does this solve?\nThe cache went stale.\n\n"
            "**Solution**\ns\n\n**Testing**\nt\n",
            0,
        ),
        ("", 1),
        ("Just a sentence about the change.\n", 3),
        ("## Problem\np\n\n## Solution\ns\n\n## Testing\nt\n", 3),
        # An ATX heading beside a correct bold one is a duplicate, not a miss.
        ("## Problem\nx\n\n" + good, 1),
        ("**Problem**\np\n\n**Solution**\ns\n", 1),
        ("**Testing**\nt\n\n**Problem**\np\n\n**Solution**\ns\n", 1),
        ("**Problem**\n\n**Solution**\ns\n\n**Testing**\nt\n", 1),
        ("**Problem**\np\n\n**Problem**\np\n\n**Solution**\ns\n\n**Testing**\nt\n", 1),
        # The template, opened and pushed.
        (
            "**Problem**\n\nWhat problem does this solve?\n\n**Solution**\n\n"
            "What changed?\n\n**Testing**\n\nWhat did you run?\n",
            3,
        ),
        # A heading inside a fence is quoted text, not a section.
        ("Here is the template:\n\n```\n" + good + "```\n", 3),
        # ...including a tilde fence, which is the other arm of the pattern.
        ("~~~\n" + good + "~~~\n", 3),
        # ...and the fence must close on its own marker, not a shorter one.
        ("````\n```\n" + good + "````\n", 3),
        # An unclosed fence runs to the end, as GitHub renders it.
        ("```\n" + good, 3),
        # Bold opening a sentence is prose.
        ("**Problem** was the cache.\n\n" + good, 0),
        # A trailing colon is the same heading.
        ("**Problem:**\np\n\n**Solution:**\ns\n\n**Testing:**\nt\n", 0),
    ]
    failures = []
    for body, want in cases:
        got = check(body)
        if len(got) != want:
            head = body.splitlines()[0] if body.splitlines() else "<empty>"
            failures.append(
                f"{head!r}: expected {want} violation(s), got {len(got)}: {got}"
            )
    return failures


def die(message: str) -> None:
    """Exit 2: no body was read, which is not the same verdict as a bad one."""
    print(f"ci/pr-body.py: {message}", file=sys.stderr)
    sys.exit(2)


def read_body(args: argparse.Namespace) -> str:
    if args.env:
        if args.env not in os.environ:
            die(f"${args.env} is not set")
        return os.environ[args.env]
    if args.path == "-":
        return sys.stdin.read()
    if args.path:
        try:
            with open(args.path, encoding="utf-8") as fh:
                return fh.read()
        except OSError as exc:
            die(str(exc))
    cmd = ["gh", "pr", "view"]
    if args.pr:
        cmd.append(str(args.pr))
    cmd += ["--json", "body", "--jq", ".body"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        die("gh is not installed; pass a file, --env or - instead")
    except subprocess.CalledProcessError as exc:
        die(f"could not read the PR body\n{exc.stderr.strip()}")
    return out.stdout


def emit(lines: list[str]) -> None:
    """Print the verdict, and repeat it in the CI run summary when there is one."""
    text = "\n".join(lines)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(f"## PR description\n\n```\n{text}\n```\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", nargs="?", help="file holding the body, or - for stdin")
    parser.add_argument(
        "--env", metavar="VAR", help="environment variable holding the body"
    )
    parser.add_argument("--pr", type=int, help="PR number to read via gh")
    args = parser.parse_args()

    broken = selftest()
    if broken:
        print("the check's own fixtures failed, so its verdict cannot be trusted:")
        for f in broken:
            print(f"  {f}")
        return 1

    problems = check(read_body(args))
    if not problems:
        emit(["PR description structure ok"])
        return 0

    report = [
        f"PR description does not follow the required structure ({len(problems)}):"
    ]
    report += [f"  {p}" for p in problems]
    report += [
        "",
        "Every description carries these three sections, in this order, with bold",
        "headings. Extra sections (Notes, Provenance, Stack, ...) are welcome",
        "anywhere.",
        "",
        SKELETON,
        "",
        "Check a description before pushing with `./ci/pr-body.py --pr <n>`.",
        "See CONTRIBUTING.md, 'Commits and pull requests'.",
    ]
    emit(report)
    return 1


if __name__ == "__main__":
    sys.exit(main())
