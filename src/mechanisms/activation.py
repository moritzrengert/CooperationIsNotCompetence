def resolve_activation(thresholds: list[int]) -> tuple[list[bool], int]:
    """Activate the largest feasible number of offers/contributions.

    `thresholds[i]` is the minimum total number of active players required by i.
    A threshold of n + 1 is therefore never active in an n-player group.
    """
    n = len(thresholds)
    active_count = max(k for k in range(n + 1) if sum(t <= k for t in thresholds) >= k)
    return [threshold <= active_count for threshold in thresholds], active_count
