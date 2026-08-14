#!/usr/bin/env python3

import re
import sys
import textwrap

import duckdb


def split_statements(sql: str):
    """Split on `;` and `.print`, ignoring both inside comments and strings."""
    out, buf, i, n = [], [], 0, len(sql)
    in_line_comment = in_block_comment = False
    quote = None
    while i < n:
        c, nxt = sql[i], sql[i + 1] if i + 1 < n else ""
        if in_line_comment:
            if c == "\n":
                in_line_comment = False
            buf.append(c)
        elif in_block_comment:
            if c == "*" and nxt == "/":
                in_block_comment = False
                buf.append("*/")
                i += 2
                continue
            buf.append(c)
        elif quote:
            buf.append(c)
            if c == quote:
                quote = None
        elif c == "-" and nxt == "-":
            in_line_comment = True
            buf.append(c)
        elif c == "/" and nxt == "*":
            in_block_comment = True
            buf.append("/*")
            i += 2
            continue
        elif c in "'\"":
            quote = c
            buf.append(c)
        elif c == ";":
            out.append("".join(buf))
            buf = []
        else:
            buf.append(c)
        i += 1
    if "".join(buf).strip():
        out.append("".join(buf))
    return out


def strip_comments(s: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    return "\n".join(re.sub(r"--.*$", "", ln) for ln in s.split("\n"))


def render(cols, rows, limit=200):
    if not rows:
        return "    (no rows)"
    w = [max(len(str(c)), *(len(str(r[j])) for r in rows[:limit]))
         for j, c in enumerate(cols)]
    w = [min(x, 34) for x in w]
    def fmt(vals):
        return "  ".join(str(v)[:w[j]].ljust(w[j]) for j, v in enumerate(vals))
    lines = ["    " + fmt(cols), "    " + "  ".join("-" * x for x in w)]
    lines += ["    " + fmt(r) for r in rows[:limit]]
    if len(rows) > limit:
        lines.append(f"    ... {len(rows) - limit} more rows")
    return "\n".join(lines)


def main(path):
    sql = open(path).read()
    con = duckdb.connect()
    print(f"# duckdb {duckdb.__version__}  script={path}\n", flush=True)

    for idx, raw in enumerate(split_statements(sql)):
        for line in raw.split("\n"):
            m = re.match(r"\s*\.print\s+'(.*)'\s*$", line)
            if m:
                print(m.group(1), flush=True)
        stmt = "\n".join(
            ln for ln in raw.split("\n")
            if not re.match(r"\s*\.print\s", ln)
        )
        if not strip_comments(stmt).strip():
            continue
        try:
            cur = con.execute(stmt)
        except Exception as exc:
            head = textwrap.shorten(" ".join(stmt.split()), 240)
            print(f"\n!! STATEMENT {idx} FAILED\n   {head}\n"
                  f"   {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            sys.exit(1)
        # only echo genuine result sets, not SET / CREATE / INSERT / COPY
        first = strip_comments(stmt).strip().split(None, 1)[0].upper()
        if first not in ("SELECT", "WITH", "DESCRIBE", "SHOW",
                         "PRAGMA", "EXPLAIN", "SUMMARIZE"):
            continue
        try:
            rows = cur.fetchall()
        except Exception:
            continue
        if cur.description and rows is not None:
            print(render([d[0] for d in cur.description], rows), flush=True)
            print(flush=True)

    print("=== script completed, all statements OK ===", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: run_sql.py <script.sql>")
    main(sys.argv[1])
