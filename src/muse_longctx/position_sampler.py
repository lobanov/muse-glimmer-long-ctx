#!/usr/bin/env python3
"""PLAN §6 — context/position sampler.

Trainer-independent component that turns pre-tokenized sequences into training samples
with controllable position layouts, for Glimmer's hybrid SWA-RoPE / NoPE-global attention.

Produces (per sample):
    input_ids, position_ids, labels, loss_mask, evidence_positions,
    physical_sequence_length, virtual_context_length, mode

Modes:
    normal          positions 0..L-1                        (baseline)
    uniform_offset  random constant offset, then 0..L-1     (absolute-position robustness)
    random_segments physical seq split into k segments; each segment gets a contiguous
                    virtual block; blocks scattered in virtual space (block-internal order
                    preserved; global monotonicity NOT guaranteed — that is the point)
    pose            PoSE-style skipped positions (Zhu et al. 2024): within-chunk positions
                    contiguous, chunk starts jump forward; monotonic; max position ≈ V-1
    yarn_random     like pose but virtual length = yarn_factor * physical (randomized
                    YaRN-style virtual ranges)
    genuine         identity — virtual == physical (mode 6 in the plan; the arm expected
                    to carry nearly all signal for this architecture)

Architecture scope note (PLAN §6): modes 2-5 train the local RoPE layers' absolute-position
robustness only; the 13 global NoPE layers have no position IDs. Implemented for ablation
completeness; do not budget majority training time to them.
"""
import random
from dataclasses import dataclass, field

MODES = ("normal", "uniform_offset", "random_segments", "pose", "yarn_random", "genuine")


@dataclass
class PositionLayout:
    mode: str
    physical_len: int
    virtual_len: int
    position_ids: list


@dataclass
class TrainingSample:
    input_ids: list
    position_ids: list
    labels: list
    loss_mask: list           # 1 where loss applies
    evidence_positions: list  # virtual positions of evidence-token spans
    physical_len: int
    virtual_len: int
    mode: str
    meta: dict = field(default_factory=dict)


def sample_positions(mode: str, physical_len: int, virtual_len: int | None = None,
                     rng: random.Random | None = None, *,
                     n_segments: int | None = None, yarn_factor: float = 4.0) -> PositionLayout:
    rng = rng or random.Random(0)
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; known: {MODES}")
    if virtual_len is None:
        virtual_len = physical_len if mode in ("normal", "genuine") else \
            int(physical_len * yarn_factor)
    if virtual_len < physical_len:
        raise ValueError(f"virtual_len {virtual_len} < physical_len {physical_len}")

    if mode in ("normal", "genuine"):
        pos = list(range(physical_len))

    elif mode == "uniform_offset":
        off = rng.randint(0, virtual_len - physical_len)
        pos = [off + i for i in range(physical_len)]

    elif mode == "random_segments":
        k = n_segments or rng.randint(2, max(2, min(8, physical_len // 2048)))
        cuts = sorted(rng.sample(range(1, physical_len), k - 1)) if k > 1 else []
        bounds = [0, *cuts, physical_len]
        seg_lens = [bounds[i + 1] - bounds[i] for i in range(k)]
        # scatter non-overlapping contiguous virtual blocks (rejection sampling;
        # largest-first). Global monotonicity is NOT guaranteed — by design.
        placed: list[tuple[int, int]] = []  # (start, end) virtual intervals
        starts: list[int | None] = [None] * k
        for i in sorted(range(k), key=lambda j: -seg_lens[j]):
            n = seg_lens[i]
            for _ in range(200):
                s = rng.randint(0, virtual_len - n)
                if all(s >= e or s + n <= b for b, e in placed):
                    starts[i] = s
                    placed.append((s, s + n))
                    break
            if starts[i] is None:
                raise ValueError("random_segments: could not place segments; "
                                 f"physical={physical_len} k={k} virtual={virtual_len} "
                                 "(need virtual_len >> physical_len * (k+1)/k)")
        pos = []
        for i in range(k):
            pos.extend(range(starts[i], starts[i] + seg_lens[i]))

    elif mode in ("pose", "yarn_random"):
        # k+1 anchors -> k chunks; chunk starts get cumulative forward jumps whose sum
        # is virtual_len - physical_len (with jitter); positions strictly increasing.
        k = n_segments or rng.randint(2, max(2, min(16, physical_len // 2048)))
        cuts = sorted(rng.sample(range(1, physical_len), k - 1)) if k > 1 else []
        bounds = [0, *cuts, physical_len]
        gap_total = virtual_len - physical_len
        weights = [rng.uniform(0.5, 1.5) for _ in range(k)]
        wsum = sum(weights)
        pos, acc = [], 0
        for i in range(k):
            seg = bounds[i + 1] - bounds[i]
            gap = int(gap_total * weights[i] / wsum) if i < k - 1 else 0
            acc += gap if i > 0 else 0
            pos.extend(range(acc, acc + seg))
            acc += seg
        # top up to hit the tail exactly (last gap absorbed rounding)
        deficit = (virtual_len - 1) - pos[-1]
        if deficit > 0 and len(pos) > 0:
            idx = bounds[-2] if k > 1 else 0
            pos = [p + deficit if i >= idx else p for i, p in enumerate(pos)]

    return PositionLayout(mode, physical_len, virtual_len, pos)


def build_training_sample(input_ids: list, loss_token_spans: list | None = None,
                          evidence_token_spans: list | None = None,
                          layout: PositionLayout | None = None,
                          **layout_kwargs) -> TrainingSample:
    """loss_token_spans: [(s,e), ...] token spans (prompt excluded → labels -100 elsewhere)."""
    L = len(input_ids)
    layout = layout or sample_positions("normal", L)
    pos = layout.position_ids
    assert len(pos) == L
    loss_mask = [0] * L
    for s, e in (loss_token_spans or []):
        for i in range(s, min(e, L)):
            loss_mask[i] = 1
    labels = [t if m else -100 for t, m in zip(input_ids, loss_mask)]
    ev = [pos[s:e] for s, e in (evidence_token_spans or [])]
    return TrainingSample(input_ids=list(input_ids), position_ids=pos, labels=labels,
                          loss_mask=loss_mask, evidence_positions=ev,
                          physical_len=layout.physical_len, virtual_len=layout.virtual_len,
                          mode=layout.mode,
                          meta=dict(layout_kwargs))


# --------------------------------------------------------------- self-test
def _selftest():
    rng = random.Random(42)
    L = 8192
    ids = list(range(L))  # stand-in token ids
    # 1) identity modes
    for m in ("normal", "genuine"):
        lay = sample_positions(m, L, rng=rng)
        assert lay.position_ids == list(range(L)) and lay.virtual_len == L
    # 2) uniform offset
    lay = sample_positions("uniform_offset", L, virtual_len=4 * L, rng=rng)
    assert lay.position_ids[0] >= 0 and lay.position_ids[-1] < 4 * L
    assert lay.position_ids[-1] - lay.position_ids[0] == L - 1
    # 3) PoSE-family: strictly increasing, hits tail, stays < virtual_len
    for m in ("pose", "yarn_random"):
        for V in (2 * L, 4 * L, 7 * L // 2):
            lay = sample_positions(m, L, virtual_len=V, rng=rng)
            p = lay.position_ids
            assert all(b > a for a, b in zip(p, p[1:])), f"{m} not strictly increasing"
            assert p[-1] == V - 1, f"{m} tail miss: {p[-1]} != {V-1}"
            assert p[0] == 0
    # 4) random_segments: block-internal contiguity, range validity, block non-overlap
    lay = sample_positions("random_segments", L, virtual_len=4 * L, rng=rng)
    p = lay.position_ids
    assert all(0 <= x < 4 * L for x in p)
    runs = []
    cur = [p[0], p[0]]
    for a, b in zip(p, p[1:]):
        if b == a + 1:
            cur[1] = b
        else:
            runs.append(tuple(cur))
            cur = [b, b]
    runs.append(tuple(cur))
    for i, (b1, e1) in enumerate(runs):
        for b2, e2 in runs[i + 1:]:
            assert b1 >= e2 or b2 >= e1, "overlapping position blocks"
    # 5) full sample build
    s = build_training_sample(ids, loss_token_spans=[(L - 100, L)],
                              evidence_token_spans=[(10, 20)],
                              layout=sample_positions("pose", L, virtual_len=4 * L, rng=rng))
    assert sum(s.loss_mask) == 100 and s.labels[:L - 100] == [-100] * (L - 100)
    assert len(s.evidence_positions[0]) == 10
    print("selftest OK: modes", MODES)


if __name__ == "__main__":
    _selftest()
