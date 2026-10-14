"""Compare every report field; allow 1e-7 absolute float drift across platforms."""
import json
import math
import sys
from pathlib import Path

def compare(a, b, path='root'):
    if isinstance(a, dict) and isinstance(b, dict):
        assert a.keys() == b.keys(), path
        for key in a: compare(a[key], b[key], path+'.'+key)
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)): compare(x,y,f'{path}[{i}]')
    elif isinstance(a,float) and isinstance(b,float):
        assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-7),(path,a,b)
    else:
        assert a==b,(path,a,b)

if __name__ == '__main__':
    actual, expected = [json.loads(Path(p).read_text()) for p in sys.argv[1:]]
    compare(actual,expected)
    print(f"PASS: {actual['summary']['records']} records; exact metadata/counts, bounded floating-point drift.")
