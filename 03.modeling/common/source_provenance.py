"""Source-only hash compatibility. Artifact hashes always remain byte-exact."""
import hashlib
from pathlib import Path


def verify_source_digest(path,recorded):
    """Accept byte identity or uniform LF/CRLF conversion; reject all other edits.

    Existing records are not rewritten. The legacy recorded digest can represent
    LF or CRLF bytes; neither whitespace nor Python syntax is normalized.
    """
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()==recorded:return 'byte_exact'
    lf=raw.replace(b'\r\n',b'\n')
    candidates=(lf,lf.replace(b'\n',b'\r\n'))
    assert any(hashlib.sha256(value).hexdigest()==recorded for value in candidates), \
        f'소스가 기록과 다름(줄바꿈 외 변경): {path}'
    return 'line_endings_equivalent'
