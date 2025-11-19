import itertools

import numpy as np
import pandas as pd

from scipy import spatial, optimize


def temporal_intersection_over_union(a: pd.DataFrame, b: pd.DataFrame, threshold: float) -> float:
    """
    Temporal intersection over union of two tracks.
    Perfect match will return 1, complete misalignment will return 0.
    If the two tracks don't overlap spatially in any frame, -infinity is returned.
    Both tracks must be a Pandas DataFrame with 'x', 'y', and ordered 'frame' columns.
    Track input order is irrelevant.

    :param a: the first track
    :param b: the second track
    :param threshold: the maximum Euclidean distance between two detections for them to be considered as overlapping
    :return: the metric value
    """
    ab, a_ind, b_ind = np.intersect1d(a.frame, b.frame, assume_unique=True, return_indices=True)
    distances = np.hypot(a.x.values[a_ind] - b.x.values[b_ind], a.y.values[a_ind] - b.y.values[b_ind])
    intersection = np.count_nonzero(distances <= threshold)

    if intersection == 0:
        return -np.inf

    union = len(a.frame) + len(b.frame) - len(ab)

    return intersection / union


_columns = ['x', 'y', 'frame', 'particle']


def mean_temporal_intersection_over_union(a: pd.DataFrame, b: pd.DataFrame, threshold: float) -> dict[str, float]:
    grouped_a = [pd.DataFrame.from_records(g, columns=_columns)
                 for _, g in itertools.groupby(a[_columns].values, lambda row: row[3])]
    grouped_b = [pd.DataFrame.from_records(g, columns=_columns)
                 for _, g in itertools.groupby(b[_columns].values, lambda row: row[3])]

    n_a, n_b = len(grouped_a), len(grouped_b)

    distances = np.zeros((n_a, n_b))
    for i, ga in enumerate(grouped_a):
        for j, gb in enumerate(grouped_b):
            distances[i, j] = 1 - temporal_intersection_over_union(ga, gb, threshold)

    a_ind, b_ind = linear_assignment(distances, 1)

    mask = (a_ind < n_a) & (b_ind < n_b)
    n_matched = np.count_nonzero(mask)
    n_unmatched = n_a + n_b - 2 * n_matched
    matched_distance = np.sum(distances[a_ind[mask], b_ind[mask]])

    mtiou = 1 - (2 * matched_distance + n_unmatched) / (n_a + n_b)
    mmtiou = 1 - matched_distance / n_matched

    fp_tracks = np.sum((a_ind >= n_a) & (b_ind < n_b))
    fn_tracks = np.sum((b_ind >= n_b) & (a_ind < n_a))

    return {'mean_tiou': mtiou, 'matched_mean_tiou': mmtiou, 'fp_tracks': fp_tracks, 'fn_tracks': fn_tracks}


# Function aliases
tiou = temporal_intersection_over_union
mtiou = mean_temporal_intersection_over_union


def mot_metrics(a: pd.DataFrame, b: pd.DataFrame, threshold: float) -> dict[str, float]:
    TP, FP, FN, IDS = 0, 0, 0, 0
    GT = len(a)
    d, c = 0, 0

    last_matched = {}

    frames = sorted(set(a.frame.unique()) | set(b.frame.unique()))
    for f in frames:
        a_at_f = a.loc[a.frame == f]
        b_at_f = b.loc[b.frame == f]

        distances = spatial.distance.cdist(a_at_f[['x', 'y']], b_at_f[['x', 'y']])

        a_ind, b_ind = linear_assignment(distances, threshold)

        mask = (a_ind < len(a_at_f)) & (b_ind < len(b_at_f))

        FP += np.sum((a_ind >= len(a_at_f)) & (b_ind < len(b_at_f)))
        FN += np.sum((b_ind >= len(b_at_f)) & (a_ind < len(a_at_f)))

        d += np.sum(distances[a_ind[mask], b_ind[mask]])
        c += len(mask)

        current_matched = {}
        for ai, bi in zip(a_ind[mask], b_ind[mask]):
            p = b_at_f.particle.iat[bi]
            matched_to = a_at_f.particle.iat[ai]

            current_matched[p] = matched_to

            if p in last_matched and last_matched[p] != matched_to:
                IDS += 1

            else:
                TP += 1

        last_matched = current_matched

    MOTA = 1 - (FN + FP + IDS) / GT
    MOTP = d / c

    return {'MOTA': MOTA, 'MOTP': MOTP, 'TP': TP, 'FN': FN, 'FP': FP, 'IDS': IDS}


def hota(gt: pd.DataFrame, tr: pd.DataFrame, threshold: float) -> dict[str, float]:
    """Slightly adapted from https://github.com/JonathonLuiten/TrackEval"""

    # Ensure particle ids are sorted from 0 to max(n)
    gt = gt.copy()
    tr = tr.copy()

    gt.particle = gt.particle.map({old: new for old, new in zip(gt.particle.unique(), range(gt.particle.nunique()))})
    tr.particle = tr.particle.map({old: new for old, new in zip(tr.particle.unique(), range(tr.particle.nunique()))})

    # Initialization
    num_gt_ids = gt.particle.nunique()
    num_tr_ids = tr.particle.nunique()

    frames = sorted(set(gt.frame.unique()) | set(tr.frame.unique()))

    potential_matches_count = np.zeros((num_gt_ids, num_tr_ids))
    gt_id_count = np.zeros((num_gt_ids, 1))
    tracker_id_count = np.zeros((1, num_tr_ids))

    HOTA_TP, HOTA_FN, HOTA_FP = 0, 0, 0
    LocA = 0.0

    # Compute similarities (inverted normalized distance)
    similarities = [1 - np.clip(spatial.distance.cdist(gt[gt.frame == t][['x', 'y']],
                                                       tr[tr.frame == t][['x', 'y']]) / threshold, 0, 1)
                    for t in frames]

    # Accumulate global track information
    for t in frames:
        gt_ids_t = gt[gt.frame == t].particle.to_numpy()
        tr_ids_t = tr[tr.frame == t].particle.to_numpy()

        similarity = similarities[t]
        sim_iou_denom = similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
        sim_iou = np.zeros_like(similarity)
        sim_iou_mask = sim_iou_denom > 0 + np.finfo('float').eps
        sim_iou[sim_iou_mask] = similarity[sim_iou_mask] / sim_iou_denom[sim_iou_mask]
        potential_matches_count[gt_ids_t[:, None], tr_ids_t[None, :]] += sim_iou

        gt_id_count[gt_ids_t] += 1
        tracker_id_count[0, tr_ids_t] += 1

    global_alignment_score = potential_matches_count / (gt_id_count + tracker_id_count - potential_matches_count)
    matches_count = np.zeros_like(potential_matches_count)

    # Calculate scores for each timestep
    for t in frames:
        gt_ids_t = gt[gt.frame == t].particle.to_numpy()
        tr_ids_t = tr[tr.frame == t].particle.to_numpy()

        if len(gt_ids_t) == 0:
            HOTA_FP += len(tr_ids_t)
            continue

        if len(tr_ids_t) == 0:
            HOTA_FN += len(gt_ids_t)
            continue

        similarity = similarities[t]
        score_mat = global_alignment_score[gt_ids_t[:, None], tr_ids_t[None, :]] * similarity

        match_rows, match_cols = optimize.linear_sum_assignment(-score_mat)

        actually_matched_mask = similarity[match_rows, match_cols] > 0
        alpha_match_rows = match_rows[actually_matched_mask]
        alpha_match_cols = match_cols[actually_matched_mask]

        num_matches = len(alpha_match_rows)

        HOTA_TP += num_matches
        HOTA_FN += len(gt_ids_t) - num_matches
        HOTA_FP += len(tr_ids_t) - num_matches

        if num_matches > 0:
            LocA += sum(similarity[alpha_match_rows, alpha_match_cols])
            matches_count[gt_ids_t[alpha_match_rows], tr_ids_t[alpha_match_cols]] += 1

    ass_a = matches_count / np.maximum(1, gt_id_count + tracker_id_count - matches_count)
    AssA = np.sum(matches_count * ass_a) / np.maximum(1, HOTA_TP)
    DetA = HOTA_TP / np.maximum(1, HOTA_TP + HOTA_FN + HOTA_FP)
    HOTA = np.sqrt(DetA * AssA)

    return {'HOTA': HOTA, 'AssA': AssA, 'DetA': DetA, 'LocA': LocA,
            'HOTA TP': HOTA_TP, 'HOTA FN': HOTA_FN, 'HOTA FP': HOTA_FP}


def linear_assignment(cost_matrix, max_cost):
    height, width = cost_matrix.shape

    top_right = np.full((height, height), np.inf)
    np.fill_diagonal(top_right, max_cost)
    bottom_left = np.full((width, width), np.inf)
    np.fill_diagonal(bottom_left, max_cost)

    # noinspection PyTypeChecker
    square_cost_matrix = np.block([
        [cost_matrix, top_right],
        [bottom_left, cost_matrix.T],
    ])

    return optimize.linear_sum_assignment(square_cost_matrix)
