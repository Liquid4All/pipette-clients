"""Per-repetition energy (E6-E2)/4 and its linear fit over a sweep's output.

Reads mac_sweep.sh or pi_sweep.sh lines.

A line carries one counter energy (`soc_all_j` on the Mac, `pmic_j` on the Pi) and,
on the Mac, the `PSTR`-derived energy beside it.
"""

import collections
import re
import sys

LINE = re.compile(
    r"(\w+) n=(\d+) r=(\d+) bench_measured_s=([\d.]+) .*?elapsed_s=([\d.]+) "
    r".*?(?:soc_all|pmic)_j=([\d.]+)(?: pstr_mean_w=[\d.]+ pstr_n=\d+ pstr_j=([\d.]+))?"
)

rows = collections.defaultdict(dict)
for line in open(sys.argv[1]):
    m = LINE.match(line)
    if not m:
        continue
    ph, n, r, ts, el, soc, pj = m.groups()
    rows[(ph, int(n))][int(r)] = {
        "ts": float(ts),
        "el": float(el),
        "soc": float(soc),
        "pj": float(pj) if pj else None,
    }


def fit(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    b /= sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    ss = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys, strict=True))
    st = sum((y - my) ** 2 for y in ys)
    return a, b, 1 - ss / st


for ph in ("decode", "prefill"):
    print(f"\n{ph}: per-rep energy = (E6-E2)/4, per-rep time = (t6-t2)/4")
    xs = []
    ys = []
    ps = []
    for (p, n), d in sorted(rows.items()):
        if p != ph or 2 not in d or 6 not in d:
            continue
        ej = (d[6]["soc"] - d[2]["soc"]) / 4
        et = (d[6]["ts"] - d[2]["ts"]) / 4
        xs.append(n)
        ys.append(ej)
        line = (
            f"  n={n:5d}  rep {et * 1000:7.0f} ms  counter {ej:7.1f} J  -> "
            f"{ej / n * 1000:6.1f} mJ/tok  {ej / et:6.1f} W"
        )
        if d[6]["pj"] is not None and d[2]["pj"] is not None:
            pj = (d[6]["pj"] - d[2]["pj"]) / 4
            ps.append(pj)
            line += f"   | PSTR-derived {pj:7.1f} J ({pj / n * 1000:5.1f} mJ/tok)"
        print(line)
    a, b, r2 = fit(xs, ys)
    print(
        f"  counter fit: E_rep = {a:.1f} J + {b * 1000:.2f} mJ/token * n   R^2={r2:.4f}"
    )
    if len(ps) == len(xs):
        a, b, r2 = fit(xs, ps)
        print(
            f"  PSTR fit: E_rep = {a:.1f} J + {b * 1000:.2f} mJ/token * n"
            f"   R^2={r2:.4f}"
        )
    overheads = []
    for n in xs:
        e2 = rows[(ph, n)][2]["soc"]
        e6 = rows[(ph, n)][6]["soc"]
        overheads.append(f"n={n}: {e2 - 2 * ((e6 - e2) / 4):.0f} J")
    print("  fixed overhead per run (E2 - 2*E_rep):", ", ".join(overheads))
