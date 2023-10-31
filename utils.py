import io
import contextlib
import sys

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
