"""Four-port equivalence partitions; no pin anchor or OCR targets."""
import itertools
import numpy as np

PAIRS = list(itertools.combinations(range(4), 2))
PORTS = ['left', 'right', 'top', 'bottom']


def partitions(n=4):
    def visit(prefix):
        if len(prefix) == n:
            yield prefix
        else:
            for value in range(max(prefix) + 2):
                yield from visit(prefix + [value])
    return list(visit([0]))


PARTITIONS = np.asarray(partitions(), np.int64)
RELATIONS = np.asarray([[row[a] == row[b] for a, b in PAIRS]
                        for row in PARTITIONS], np.float32)


def port_membership(segments, point, radius=6, tolerance=1.5):
    """Labels only: ambiguous or absent annotation ports remain unsupervised."""
    x, y = point
    owners = []
    for px, py in [(x-radius, y), (x+radius, y), (x, y-radius), (x, y+radius)]:
        found = set()
        for group, a, b in segments:
            a, b = np.asarray(a, float), np.asarray(b, float)
            delta = b-a
            length = np.dot(delta, delta)
            if not length:
                continue
            t = np.clip(np.dot(np.array([px, py])-a, delta)/length, 0, 1)
            if np.linalg.norm(np.array([px, py])-(a+t*delta)) <= tolerance:
                found.add(group)
        owners.append(next(iter(found)) if len(found) == 1 else None)
    target = np.array([float(owners[a] == owners[b]) for a, b in PAIRS], np.float32)
    valid = np.array([owners[a] is not None and owners[b] is not None
                      for a, b in PAIRS], np.float32)
    return target, valid, owners


def compatible_partitions(target, valid):
    return np.all((RELATIONS == target) | ~valid.astype(bool), axis=1)


def transitivity_violations(edges):
    lookup = {pair: i for i, pair in enumerate(PAIRS)}
    return sum(sum(bool(edges[lookup[tuple(sorted(pair))]]) for pair in
                   itertools.combinations(triple, 2)) == 2
               for triple in itertools.combinations(range(4), 3))
