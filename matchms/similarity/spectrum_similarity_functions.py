from typing import List, Tuple
import numba
import numpy as np


# @numba.njit
def collect_peak_pairs(spec1: np.ndarray, spec2: np.ndarray,
                       tolerance: float, shift: float = 0, mz_power: float = 0.0,
                       intensity_power: float = 1.0):
    # pylint: disable=too-many-arguments
    """Find matching pairs between two spectra.

    Args
    ----
    spec1:
        Spectrum peaks and intensities as numpy array.
    spec2:
        Spectrum peaks and intensities as numpy array.
    tolerance
        Peaks will be considered a match when <= tolerance appart.
    shift
        Shift spectra peaks by shift. The default is 0.
    mz_power:
        The power to raise mz to in the cosine function. The default is 0, in which
        case the peak intensity products will not depend on the m/z ratios.
    intensity_power:
        The power to raise intensity to in the cosine function. The default is 1.

    Returns
    -------
    matching_pairs : numpy array
        Array of found matching peaks.
    """
    matches = np.abs(spec1[:, 0, None] - (spec2[:, 0] + shift)) <= tolerance
    idx1, idx2 = np.nonzero(matches)
    if len(idx1) == 0:
        return None
    pprod_spec1 = spec1[idx1, 0] ** mz_power * spec1[idx1, 1] ** intensity_power
    pprod_spec2 = spec2[idx2, 0] ** mz_power * spec2[idx2, 1] ** intensity_power
    return np.stack([idx1, idx2, pprod_spec1 * pprod_spec2], -1)

@numba.njit
def find_matches(spec1_mz: np.ndarray, spec2_mz: np.ndarray,
                 tolerance: float, shift: float = 0) -> List[Tuple[int, int]]:
    """Faster search for matching peaks.
    Makes use of the fact that spec1 and spec2 contain ordered peak m/z (from
    low to high m/z).

    Parameters
    ----------
    spec1_mz:
        Spectrum peak m/z values as numpy array. Peak mz values must be ordered.
    spec2_mz:
        Spectrum peak m/z values as numpy array. Peak mz values must be ordered.
    tolerance
        Peaks will be considered a match when <= tolerance appart.
    shift
        Shift peaks of second spectra by shift. The default is 0.

    Returns
    -------
    matches
        List containing entries of type (idx1, idx2).

    """
    lowest_idx = 0
    matches = []
    for peak1_idx in range(spec1_mz.shape[0]):
        mz = spec1_mz[peak1_idx]
        low_bound = mz - tolerance
        high_bound = mz + tolerance
        for peak2_idx in range(lowest_idx, spec2_mz.shape[0]):
            mz2 = spec2_mz[peak2_idx] + shift
            if mz2 > high_bound:
                break
            if mz2 < low_bound:
                lowest_idx = peak2_idx + 1
            else:
                matches.append((peak1_idx, peak2_idx))
    return matches


@numba.njit(fastmath=True)
def score_best_matches(matching_pairs: np.ndarray, spec1: np.ndarray,
                       spec2: np.ndarray, mz_power: float = 0.0,
                       intensity_power: float = 1.0) -> Tuple[float, int]:
    """Calculate cosine-like score by multiplying matches. Does require a sorted
    list of matching peaks (sorted by intensity product)."""
    score = float(0.0)
    used_matches = int(0)
    used1 = set()
    used2 = set()
    for i in range(matching_pairs.shape[0]):
        if not matching_pairs[i, 0] in used1 and not matching_pairs[i, 1] in used2:
            score += matching_pairs[i, 2]
            used1.add(matching_pairs[i, 0])  # Every peak can only be paired once
            used2.add(matching_pairs[i, 1])  # Every peak can only be paired once
            used_matches += 1

    # Normalize score:
    spec1_power = spec1[:, 0] ** mz_power * spec1[:, 1] ** intensity_power
    spec2_power = spec2[:, 0] ** mz_power * spec2[:, 1] ** intensity_power

    score = score/(np.sum(spec1_power ** 2) ** 0.5 * np.sum(spec2_power ** 2) ** 0.5)
    return score, used_matches

def number_matching(numbers_1, numbers_2, tolerance):
    """Find all pairs between numbers_1 and numbers_2 which are within tolerance.
    """
    matching = np.abs(numbers_1[:, None] - numbers_2[None, :]) <= tolerance
    rows, cols = np.nonzero(matching)
    return rows, cols, np.full_like(rows, True)

def number_matching_symmetric(numbers_1, tolerance):
    """Find all pairs between numbers_1 and numbers_1 which are within tolerance.
    """
    matching = np.abs(numbers_1[:, None] - numbers_1[None, :]) <= tolerance
    rows, cols = np.nonzero(matching)
    return rows, cols, np.full_like(rows, True)

def number_matching_ppm(numbers_1, numbers_2, tolerance_ppm):
    """Find all pairs between numbers_1 and numbers_2 which are within tolerance.
    """
    pairwise_diff = numbers_1[:, None] - numbers_2[None, :]
    pairwise_mean = numbers_1[:, None]*.5 + numbers_2[None, :]*.5
    pairwise_ppm = np.abs(pairwise_diff)/pairwise_mean * 1e6 <= tolerance_ppm
    rows, cols = np.nonzero(pairwise_ppm)
    return rows, cols, np.full_like(rows, True)

def number_matching_symmetric_ppm(numbers_1, tolerance_ppm):
    """Find all pairs between numbers_1 and numbers_1 which are within tolerance.
    """
    pairwise_diff = numbers_1[:, None] - numbers_1[None, :]
    pairwise_mean = numbers_1[:, None]*.5 + numbers_1[None, :]*.5
    pairwise_ppm = np.abs(pairwise_diff)/pairwise_mean * 1e6 <= tolerance_ppm
    rows, cols = np.nonzero(pairwise_ppm)
    return rows, cols, np.full_like(rows, True)
