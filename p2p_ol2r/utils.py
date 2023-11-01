import io
import contextlib
import sys
import numpy as np

def split(data: io.BytesIO, size: int) -> list[bytes]:
    """
    Split a BytesIO object into chunks of size `size`.
    """
    data.seek(0)
    chunks = []
    while True:
        chunk = data.read(size)
        if not chunk:
            break
        chunks.append(chunk)
    return chunks

def fmt(s: str, *fmts: str) -> str:
    """
    Colorize a string.
    """
    format_defs = {
        'green': '\033[92m',
        'blue': '\033[94m',
        'purple': '\033[95m',
        'gray': '\033[90m',
        'yellow': '\033[93m',
        'italic': '\033[3m',
        'bold': '\033[1m',
    }
    return ''.join(format_defs[fmt] for fmt in fmts) + s + '\033[0m'

@contextlib.contextmanager
def silence():
    """
    Suppress stdout.
    """
    save_stdout = sys.stdout
    sys.stdout = io.StringIO()
    yield
    sys.stdout = save_stdout

def ndcg(expected: list[str], actual: list[str], form="linear"):
    """
    Calculate normalized discounted cumulative gain (NDCG).
    
    Args:
        expected: list of expected result IDs
        actual: list of actual result IDs
        form: formula to use, either 'linear' or 'exp'
    """
    rel_pred = [(len(expected) - actual.index(id)) for id in expected]
    rel_true = list(range(len(expected), 0, -1))
    p = len(rel_pred)
    discount = 1 / (np.log2(np.arange(p) + 2))

    if form == "linear":
        idcg = np.sum(rel_true[:p] * discount)
        dcg = np.sum(rel_pred[:p] * discount)
    elif form == "exp":
        idcg = np.sum([2**x - 1 for x in rel_true[:p]] * discount)
        dcg = np.sum([2**x - 1 for x in rel_pred[:p]] * discount)
    else:
        raise ValueError("Only supported for two formula, 'linear' or 'exp'")

    return dcg / idcg
