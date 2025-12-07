import random

import numpy as np


def shift_absorb_grow(
        data: np.ndarray,
        k: int | None = None,
        radius: float = 0.3,
        iterations: int = 100,
        tol: float = 1e-2,
        min_points: int = 7,
        seed: int = 0
):
    # TODO: adapt r in higher dimensions
    np.random.seed(seed)
    random.seed(seed)
    n_points = len(data)
    if k:
        k = min(max(k, 1), n_points)
    else:
        k = 15

    centroids = data[random.sample(range(n_points), k=k)]
    radii = np.array([radius] * k)
    frozen = np.zeros(k, dtype=bool)

    def absorb_grow() -> None:
        nonlocal centroids, radii, frozen
        c_idx = 0
        while c_idx < (n := len(centroids)):
            c = centroids[c_idx]
            r = radii[c_idx]
            c_others = np.concat([centroids[:c_idx], centroids[c_idx + 1:]])
            r_others = np.concat([radii[:c_idx], radii[c_idx + 1:]])
            dist = np.sqrt(((c_others - c) ** 2).sum(1))  # (n - 1,)
            enclose = dist + r < r_others + tol  # some circle(s) fully enclose current one
            if any(enclose):  # absorbed
                # Remove current circle:
                centroids = c_others
                radii = r_others
                frozen = np.concat([frozen[:c_idx], frozen[c_idx + 1:]])
                continue  # go to next circle in "centroids"

            enclosed = dist + r_others < r + tol  # some circle(s) are fully enclosed by current circle
            if any(enclosed):  # absorbs
                keep = np.ones(n, dtype=bool)
                keep[np.arange(n) != c_idx] = ~enclosed
                c_idx = int(sum(keep[:c_idx]))
                centroids = centroids[keep]
                radii = radii[keep]
                frozen = frozen[keep]

                c_others = c_others[~enclosed]
                r_others = r_others[~enclosed]
                dist = dist[~enclosed]

            contain = dist < r_others  # some circle(s) contain the center of current circle
            contained = dist < r  # some circle(s) are contained by current circle
            parent = contain & ~contained  # some circle(s) contain but are NOT contained by current circle
            if any(parent):
                search_mask = parent
            elif any(contain):
                search_mask = contain
            else:
                search_mask = contained

            if any(search_mask):  # grow
                vals = dist[search_mask] + r_others[search_mask]
                max_idx = vals.argmax()
                r_new = (vals[max_idx] + r) / 2.
                centroids[c_idx] = c + (r_new - r) / (dist[search_mask][max_idx] + 1e-8) * (c_others[search_mask][max_idx] - c)
                radii[c_idx] = r_new

                # Remove the circle used for growing:
                global_idx = np.arange(len(search_mask))[search_mask][max_idx]
                if global_idx < c_idx:  # if removed index is to left of "c_idx", compensate the index drift
                    c_idx -= 1
                else:
                    global_idx += 1  # because it was pushed to the left when "c_idx" was removed

                centroids = np.concat([centroids[:global_idx], centroids[global_idx + 1:]])
                radii = np.concat([radii[:global_idx], radii[global_idx + 1:]])
                frozen = np.concat([frozen[:global_idx], frozen[global_idx + 1:]])

                continue

            c_idx += 1

    def result():
        print(f"Discovered {len(centroids)} number of clusters!")
        gap = np.sqrt(((data[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)) - radii[None, :]
        cluster_pos = gap.argmin(axis=1)
        return np.column_stack((data, cluster_pos))

    for _ in range(iterations):
        i = 0
        while i < len(centroids):
            if frozen[i]:
                i += 1
                continue

            dist = ((data - centroids[i]) ** 2).sum(1)  # (n_points,)
            mask = dist < radii[i] ** 2
            mu = data[mask].mean(0)
            if np.sqrt(((mu - centroids[i]) ** 2).sum()) > tol:
                centroids[i] = mu
                i += 1
            else:
                if int(mask.sum()) > min_points:
                    frozen[i] = True
                    if all(frozen):
                        return result()
                    i += 1
                else:
                    centroids = np.concat([centroids[:i], centroids[i + 1:]])
                    radii = np.concat([radii[:i], radii[i + 1:]])
                    frozen = np.concat([frozen[:i], frozen[i + 1:]])

        absorb_grow()  # make this per centroid in the inner loop?

    return result()
