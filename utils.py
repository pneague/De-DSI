import torch
import io

def deserialize_model(model: bytes):
    buffer = io.BytesIO(model)
    return torch.load(buffer)

def split(data: io.BytesIO, size: int) -> list[bytes]:
    data.seek(0)
    chunks = []
    while True:
        chunk = data.read(size)
        if not chunk:
            break
        chunks.append(chunk)
    return chunks