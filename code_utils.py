"""Sandboxed execution + answer extraction for HumanEval / MBPP (pure python, no torch).

WARNING: this runs model-generated code in a subprocess (temp dir, CPU/memory limits, a few
destructive os/shutil calls disabled). It is NOT a hardened sandbox; run on a machine you are
comfortable with. Self-test:  python code_utils.py
"""
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

GUARD = (
    "import os, shutil, subprocess\n"
    "for _n in ('system','remove','removedirs','rmdir','unlink','rename','kill','killpg','fork','forkpty'):\n"
    "    setattr(os, _n, None)\n"
    "shutil.rmtree = None\nsubprocess.Popen = None\n\n"
)


def extract_code(text):
    blocks = re.findall(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", text, re.S)
    if blocks:
        for b in blocks:
            if "def " in b:
                return b
        return blocks[0]
    m = re.search(r"```(?:python|py|Python)?[ \t]*\n(.*)", text, re.S)   # unterminated fence
    return m.group(1) if m else text


def humaneval_question(item):
    return ("Complete the following Python function. Return the complete function in a ```python code block.\n\n"
            "```python\n" + item["prompt"] + "```")


def mbpp_question(item):
    return item["prompt"] + "\nYour code should pass these tests:\n" + "\n".join(item["test_list"])


def humaneval_program(item, completion):
    code = extract_code(completion)
    imports = "\n".join(l for l in item["prompt"].splitlines() if l.startswith(("import ", "from ")))
    if f"def {item['entry_point']}" in code:
        code = code.lstrip()          # answers start with ' ' after 'Answer:' -> IndentationError otherwise
    else:                             # completion-style fallback: keep body indentation as is
        code = item["prompt"] + code
    return GUARD + imports + "\n" + code + "\n\n" + item["test"] + f"\n\ncheck({item['entry_point']})\n"


def mbpp_program(item, completion):
    code = extract_code(completion).lstrip()
    return GUARD + "\n".join(item.get("test_imports") or []) + "\n" + code + "\n\n" + "\n".join(item["test_list"]) + "\n"


def _limits():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))


def run_program(prog, timeout=10):
    """-> (passed: bool, message: str)"""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "prog.py")
        with open(path, "w") as f:
            f.write(prog)
        try:
            r = subprocess.run([sys.executable, "-I", path], cwd=td, capture_output=True, text=True,
                               timeout=timeout, preexec_fn=_limits)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        if r.returncode == 0:
            return True, "passed"
        return False, (r.stderr or "").strip().splitlines()[-1][:200] if r.stderr else f"exit {r.returncode}"


def run_many(progs, workers=8, timeout=10):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda p: run_program(p, timeout), progs))


if __name__ == "__main__":
    he = {"prompt": "from typing import List\n\ndef add(a: int, b: int) -> int:\n    \"\"\"add\"\"\"\n",
          "entry_point": "add", "test": "def check(f):\n    assert f(1, 2) == 3\n    assert f(-1, 1) == 0\n"}
    mb = {"prompt": "Write a function to square.", "test_imports": [], "test_list": ["assert sq(3) == 9", "assert sq(-2) == 4"]}
    good = "Here:\n```python\ndef add(a, b):\n    return a + b\n```\nDone."
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    body_only = "    return a + b\n"                       # completion-style fallback
    unterminated = "```python\ndef add(a, b):\n    return a + b\n"
    loop = "```python\ndef add(a, b):\n    while True: pass\n```"
    danger = "```python\nimport os\ndef add(a, b):\n    os.system('echo hacked')\n    return a + b\n```"
    spaced = " def sq(x):\n    return x*x\n"                      # no fence, leading space (what the base model emitted)
    cases = [("mbpp no-fence leading space", mbpp_program(mb, spaced), True), ("he good", humaneval_program(he, good), True), ("he wrong", humaneval_program(he, bad), False),
             ("he body-only", humaneval_program(he, body_only), True),
             ("he unterminated fence", humaneval_program(he, unterminated), True),
             ("he infinite loop", humaneval_program(he, loop), False),
             ("he os.system blocked", humaneval_program(he, danger), False),
             ("mbpp good", mbpp_program(mb, "```python\ndef sq(x):\n    return x*x\n```"), True),
             ("mbpp wrong", mbpp_program(mb, "```python\ndef sq(x):\n    return x+x\n```"), False)]
    res = run_many([c[1] for c in cases], timeout=5)
    ok = True
    for (name, _, want), (got, msg) in zip(cases, res):
        flag = "OK " if got == want else "BAD"
        ok &= got == want
        print(f"{flag} {name:24s} passed={got} ({msg})")
    print("SELF-TEST", "PASSED" if ok else "FAILED")