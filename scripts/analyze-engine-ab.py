#!/usr/bin/env python3
"""Compare the two arms of an android-engine-ab.sh run as a Markdown table.

Spread is the coefficient of variation (stddev / mean); the delta is arm B
relative to arm A.
"""

import glob
import math
import os
import re
import sys
from collections import defaultdict


def parse_results_file(path):
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "[HEADLESS] result" not in line:
                continue
            fields = dict(re.findall(r"(\w+)=(\S+)", line))
            if "benchmark" in fields and "metric" in fields:
                try:
                    metric_val = float(fields["metric"])
                    results.append(
                        {
                            "benchmark": fields["benchmark"],
                            "type": fields.get("type", ""),
                            "model": fields.get("model", ""),
                            "quant": fields.get("quant", ""),
                            "metric": metric_val,
                            "unit": fields.get("unit", ""),
                        }
                    )
                except ValueError:
                    continue
    return results


def stats(values):
    if not values:
        return 0.0, 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) > 1:
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        stddev = math.sqrt(variance)
    else:
        stddev = 0.0
    spread_pct = (stddev / mean * 100.0) if mean != 0 else 0.0
    return mean, stddev, spread_pct


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze-engine-ab.py <output_dir>", file=sys.stderr)
        sys.exit(1)

    output_dir = sys.argv[1]
    if not os.path.isdir(output_dir):
        print(f"Error: Directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    pass_dirs = sorted(glob.glob(os.path.join(output_dir, "pass_*")))
    if not pass_dirs:
        print(f"No pass directories found in {output_dir}", file=sys.stderr)
        sys.exit(1)

    # data[arm][(benchmark, unit)] -> metric values across passes
    data = defaultdict(lambda: defaultdict(list))

    for pdir in pass_dirs:
        dirname = os.path.basename(pdir)
        # pass_<N>_<arm label>; the label itself contains underscores.
        parts = dirname.split("_", 2)
        if len(parts) < 3:
            continue
        arm_label = parts[2]

        results_file = os.path.join(pdir, "results.txt")
        if not os.path.exists(results_file):
            continue

        entries = parse_results_file(results_file)
        for e in entries:
            key = (e["benchmark"], e["unit"])
            data[arm_label][key].append(e["metric"])

    arms = sorted(data.keys())
    if len(arms) < 2:
        print(f"Found arms: {arms}. Need at least 2 arms to compare.")
        for arm in arms:
            print(f"\nArm: {arm}")
            for (bench, unit), vals in sorted(data[arm].items()):
                m, s, sp = stats(vals)
                print(
                    f"  {bench}: mean={m:.3f} {unit}, stddev={s:.3f}, "
                    f"spread={sp:.2f}% (N={len(vals)})"
                )
        return

    arm_a = arms[0]
    arm_b = arms[1]

    all_keys = sorted(set(data[arm_a].keys()) | set(data[arm_b].keys()))

    print("\n## Android Engine A/B Comparison Report")
    print(f"- Output Directory: `{output_dir}`")
    print(f"- Arm A: `{arm_a}` (N={len(data[arm_a][all_keys[0]]) if all_keys else 0})")
    print(
        f"- Arm B: `{arm_b}` (N={len(data[arm_b][all_keys[0]]) if all_keys else 0})\n"
    )

    header = (
        "| Benchmark | Unit | Arm A Mean +/- StdDev | Arm A Spread | "
        "Arm B Mean +/- StdDev | Arm B Spread | Between-Arm Delta |"
    )
    print(header)
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for bench, unit in all_keys:
        vals_a = data[arm_a].get((bench, unit), [])
        vals_b = data[arm_b].get((bench, unit), [])
        ma, sa, spa = stats(vals_a)
        mb, sb, spb = stats(vals_b)

        if ma != 0:
            delta_pct = ((mb - ma) / ma) * 100.0
            delta_str = f"{delta_pct:+.2f}%"
        else:
            delta_str = "N/A"

        row = (
            f"| `{bench}` | {unit} | {ma:.2f} +/- {sa:.2f} | {spa:.2f}% | "
            f"{mb:.2f} +/- {sb:.2f} | {spb:.2f}% | **{delta_str}** |"
        )
        print(row)
    print()


if __name__ == "__main__":
    main()
