"""Validate each distinct natural-code response outside serving measurements."""

import argparse
import ast
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import tempfile

CHECK = r"""
import dataclasses, importlib.util, itertools, json, random, sys
p=sys.argv[1];spec=importlib.util.spec_from_file_location('candidate',p);m=importlib.util.module_from_spec(spec);sys.modules['candidate']=m;spec.loader.exec_module(m)
J=m.Job; f=m.optimal_schedule
assert dataclasses.is_dataclass(J) and J.__dataclass_params__.frozen
rng=random.Random(41011);assertions=0

def check(rows):
 global assertions
 objects=[J(*r) for r in rows];original=[(j.start,j.end,j.weight) for j in objects]
 order=sorted(range(len(rows)),key=lambda i:(rows[i][1],rows[i][0],i))
 best=0;bestmask=0
 for mask in range(1<<len(rows)):
  ids=[order[k] for k in range(len(rows)) if mask>>k&1]
  if any(rows[i][1]>rows[j][0] for i,j in zip(ids,ids[1:])):continue
  value=sum(rows[i][2] for i in ids)
  if value>best or (value==best and mask<bestmask):best=value;bestmask=mask
 expected=[order[k] for k in range(len(rows)) if bestmask>>k&1]
 got=f(x for x in objects)
 assert got==(best,expected),(rows,got,(best,expected))
 assert [(j.start,j.end,j.weight) for j in objects]==original
 assertions+=2
for rows in [[],[(0,1,-1)],[(0,1,0)],[(0,2,2),(0,1,1),(1,2,1)],[(0,1,1),(1,2,2)],[(0,3,3),(0,2,3),(2,3,0)]]:check(rows)
for _ in range(250):
 rows=[]
 for _ in range(rng.randrange(9)):
  start=rng.randrange(-3,10);rows.append((start,start+rng.randrange(1,6),rng.randrange(-3,10)))
 check(rows)
for row in [(0,0,1),(2,1,9)]:
 try:f([J(*row)])
 except ValueError:pass
 else:raise AssertionError('Expected ValueError')
 assertions+=1
print(json.dumps({'passed':True,'assertions':assertions}))
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=pathlib.Path)
    p.add_argument("--output", type=pathlib.Path, required=True)
    a = p.parse_args()
    unique = {}
    total = 0
    for f in sorted(a.root.glob("c*-r*.json")):
        for row in json.loads(f.read_text())["requests"]:
            total += 1
            text = row["generated_text"]["content"]
            sha = hashlib.sha256(text.encode()).hexdigest()
            d = unique.setdefault(
                sha,
                {
                    "text": text,
                    "occurrences": 0,
                    "finish_reasons": set(),
                    "source": f.name,
                },
            )
            d["occurrences"] += 1
            d["finish_reasons"].update(row["finish_reasons"])
    for sha, d in unique.items():
        d["finish_reasons"] = sorted(d["finish_reasons"])
        text = d.pop("text")
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
        code = blocks[0] if len(blocks) == 1 else text
        try:
            assert "length" not in d["finish_reasons"], "Truncated completion"
            tree = ast.parse(code)
            for node in tree.body:
                assert isinstance(
                    node,
                    (
                        ast.Import,
                        ast.ImportFrom,
                        ast.ClassDef,
                        ast.FunctionDef,
                        ast.AnnAssign,
                        ast.Assign,
                        ast.Expr,
                    ),
                ), type(node).__name__
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    mods = (
                        [node.module]
                        if isinstance(node, ast.ImportFrom)
                        else [x.name for x in node.names]
                    )
                    assert all(
                        x
                        in (
                            "dataclasses",
                            "typing",
                            "bisect",
                            "collections",
                            "collections.abc",
                            "__future__",
                        )
                        for x in mods
                    ), mods
            with tempfile.TemporaryDirectory(prefix="dsv41-code-check-") as td:
                f = pathlib.Path(td) / "candidate.py"
                f.write_text(code)
                r = subprocess.run(
                    [sys.executable, "-I", "-S", "-c", CHECK, str(f)],
                    env={},
                    cwd=td,
                    text=True,
                    capture_output=True,
                    timeout=10,
                )
                assert r.returncode == 0, r.stderr[-2000:]
                d.update(json.loads(r.stdout))
        except Exception as e:
            d.update(passed=False, error=repr(e))
    result = {
        "requests": total,
        "distinct_outputs": len(unique),
        "passed": all(x["passed"] for x in unique.values()),
        "outputs": unique,
    }
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
