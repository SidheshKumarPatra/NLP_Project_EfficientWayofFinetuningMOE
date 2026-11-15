"""Answer extraction / matching for GSM8K and MATH (pure python, no torch)."""
import re

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def last_boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    k = j
    while k < len(text) and depth:
        depth += (text[k] == "{") - (text[k] == "}")
        k += 1
    return text[j:k - 1] if depth == 0 else None


def extract_pred(text):
    """MetaMath style 'The answer is: X' > \\boxed{X} > last number."""
    m = re.findall(r"[Tt]he answer is:?\s*(.+)", text)
    if m:
        return m[-1].strip()
    b = last_boxed(text)
    if b is not None:
        return b.strip()
    nums = _NUM.findall(text)
    return nums[-1] if nums else ""


def to_number(s):
    s = s.replace("$", "").replace("\\$", "")
    m = _NUM.search(s)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def gsm_gold(answer_field):
    return answer_field.split("####")[-1].strip().replace(",", "")


def gsm_correct(pred, gold):
    a, b = to_number(pred), to_number(gold)
    return a is not None and b is not None and abs(a - b) < 1e-4


def norm_math(s):
    s = s.strip()
    for a in ("\\left", "\\right", "\\!", "\\,", "\\;", "\\ ", "$", "\\displaystyle"):
        s = s.replace(a, "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\^\{?\\circ\}?", "", s)
    s = s.replace("\\%", "").replace("%", "")
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = s.replace(" ", "").rstrip(".")
    return s


def math_correct(pred, gold):
    p, g = norm_math(pred), norm_math(gold)
    if p == g:
        return True
    a, b = to_number(p), to_number(g)
    return (a is not None and b is not None and p.lstrip("-").replace(",", "").replace(".", "").isdigit()
            and g.lstrip("-").replace(",", "").replace(".", "").isdigit() and abs(a - b) < 1e-4)