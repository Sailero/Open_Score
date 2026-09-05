"""Content-addressed completed units; crash-safe atomic writes."""
from __future__ import annotations
import gzip
import hashlib
import json
import os
from pathlib import Path


def canonical(data):
    return json.dumps(data,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf8")


def digest(data):
    return hashlib.sha256(canonical(data)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda:stream.read(1024*1024),b""):
            h.update(data)
    return h.hexdigest()


def atomic_json(path,data):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(canonical(data))
        stream.write(b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary,path)


def write_unit(path,input_hash,result):
    envelope = {"input_hash":input_hash,"result_sha256":digest(result),"result":result}
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+f".{os.getpid()}.tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="",fileobj=raw,mode="wb",mtime=0) as compressed:
            compressed.write(canonical(envelope))
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary,path)


def read_unit(path,input_hash=None):
    try:
        with gzip.open(path,"rt",encoding="utf8") as stream:
            value = json.load(stream)
        if input_hash is not None and value["input_hash"] != input_hash:
            return None
        if value["result_sha256"] != digest(value["result"]):
            return None
        return value["result"]
    except (OSError,ValueError,TypeError,KeyError,EOFError):
        return None
