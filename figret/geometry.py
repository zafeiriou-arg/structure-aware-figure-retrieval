import math


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b):
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def frac_of_smaller(a, b):
    small = min(area(a), area(b))
    return inter(a, b) / small if small else 0.0


def frac_of_a(a, b):
    return inter(a, b) / area(a) if area(a) else 0.0


def keep_larger(boxes, thr=0.80):
    order = sorted(range(len(boxes)), key=lambda i: -area(boxes[i]))
    kept = []
    for i in order:
        if all(frac_of_smaller(boxes[i], boxes[j]) <= thr for j in kept):
            kept.append(i)
    return sorted(kept)


def edge_distance(a, b):
    dx = max(0.0, b[0] - a[2], a[0] - b[2])
    dy = max(0.0, b[1] - a[3], a[1] - b[3])
    return math.hypot(dx, dy)


def nearest_caption(fig, captions, tie_px=20):
    if not captions:
        return None
    d = [edge_distance(fig, c) for c in captions]
    best = min(d)
    close = [i for i in range(len(captions)) if d[i] <= best + tie_px]
    below = [i for i in close if captions[i][1] >= fig[3] - tie_px]
    return min(below or close, key=lambda i: d[i])
