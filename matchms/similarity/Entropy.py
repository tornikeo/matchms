import logging
from typing import Tuple
import numpy as np
from matchms.filtering.metadata_processing.add_precursor_mz import \
    _convert_precursor_mz
from matchms.typing import SpectrumType
from .BaseSimilarity import BaseSimilarity
from .spectrum_similarity_functions import (collect_peak_pairs,
                                            score_best_matches)


logger = logging.getLogger("matchms")


class Entropy(BaseSimilarity):
    # Set key characteristics as class attributes
    is_commutative = True
    # Set output data type, e.g. ("score", "float") or [("score", "float"), ("matches", "int")]
    score_datatype = [("score", np.float64), ("matches", "int")]

    def __init__(self, tolerance: float = 0.1, mz_power: float = 0.0,
                 intensity_power: float = 1.0):
        """
        Parameters
        ----------
        tolerance:
            Peaks will be considered a match when <= tolerance apart. Default is 0.1.
        mz_power:
            The power to raise mz to in the cosine function. The default is 0, in which
            case the peak intensity products will not depend on the m/z ratios.
        intensity_power:
            The power to raise intensity to in the cosine function. The default is 1.
        """
        self.tolerance = tolerance
        self.mz_power = mz_power
        self.intensity_power = intensity_power

    def pair(self, reference: SpectrumType, query: SpectrumType) -> Tuple[float, int]:
        """Calculate modified cosine score between two spectra.

        Parameters
        ----------
        reference
            Single reference spectrum.
        query
            Single query spectrum.

        Returns
        -------

        Tuple with cosine score and number of matched peaks.
        """
        def get_valid_precursor_mz(spectrum):
            """Extract valid precursor_mz from spectrum if possible. If not raise exception."""
            message_precursor_missing = \
                "Precursor_mz missing. Apply 'add_precursor_mz' filter first."
            message_precursor_no_number = \
                "Precursor_mz must be of type int or float. Apply 'add_precursor_mz' filter first."
            message_precursor_below_0 = "Expect precursor to be positive number." \
                                        "Apply 'require_precursor_mz' first"

            precursor_mz = spectrum.get("precursor_mz", None)
            if not isinstance(precursor_mz, (int, float)):
                logger.warning(message_precursor_no_number)
            precursor_mz = _convert_precursor_mz(precursor_mz)
            assert precursor_mz is not None, message_precursor_missing
            assert precursor_mz > 0, message_precursor_below_0
            return precursor_mz

        def get_matching_pairs():
            """Find all pairs of peaks that match within the given tolerance."""
            zero_pairs = collect_peak_pairs(spec1, spec2, self.tolerance, shift=0.0,
                                            mz_power=self.mz_power,
                                            intensity_power=self.intensity_power)
            precursor_mz_ref = get_valid_precursor_mz(reference)
            precursor_mz_query = get_valid_precursor_mz(query)

            mass_shift = precursor_mz_ref - precursor_mz_query
            nonzero_pairs = collect_peak_pairs(spec1, spec2, self.tolerance, shift=mass_shift,
                                               mz_power=self.mz_power,
                                               intensity_power=self.intensity_power)

            if zero_pairs is None:
                zero_pairs = np.zeros((0, 3))
            if nonzero_pairs is None:
                nonzero_pairs = np.zeros((0, 3))
            matching_pairs = np.concatenate((zero_pairs, nonzero_pairs), axis=0)
            if matching_pairs.shape[0] > 0:
                matching_pairs = matching_pairs[np.argsort(matching_pairs[:, 2])[::-1], :]
            return matching_pairs

        spec1 = reference.peaks.to_numpy
        spec2 = query.peaks.to_numpy
        matching_pairs = get_matching_pairs()
        if matching_pairs.shape[0] == 0:
            return np.asarray((float(0), 0), dtype=self.score_datatype)
        score = score_best_matches(matching_pairs, spec1, spec2,
                                   self.mz_power, self.intensity_power)
        return np.asarray(score, dtype=self.score_datatype)
