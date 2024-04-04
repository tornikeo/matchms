from typing import Tuple
import numpy as np
import numba
from numba import njit, prange
from matchms.typing import SpectrumType, List
from .BaseSimilarity import BaseSimilarity
from sparsestack import StackedSparseArray
from .spectrum_similarity_functions import (collect_peak_pairs,
                                            score_best_matches, cosine_greedy_kernel)


class CosineGreedy(BaseSimilarity):
    """Calculate 'cosine similarity score' between two spectra.

    The cosine score aims at quantifying the similarity between two mass spectra.
    The score is calculated by finding best possible matches between peaks
    of two spectra. Two peaks are considered a potential match if their
    m/z ratios lie within the given 'tolerance'.
    The underlying peak assignment problem is here solved in a 'greedy' way.
    This can perform notably faster, but does occasionally deviate slightly from
    a fully correct solution (as with the Hungarian algorithm, see
    :class:`~matchms.similarity.CosineHungarian`). In practice this will rarely
    affect similarity scores notably, in particular for smaller tolerances.

    For example

    .. testcode::

        import numpy as np
        from matchms import Spectrum
        from matchms.similarity import CosineGreedy

        reference = Spectrum(mz=np.array([100, 150, 200.]),
                             intensities=np.array([0.7, 0.2, 0.1]))
        query = Spectrum(mz=np.array([100, 140, 190.]),
                         intensities=np.array([0.4, 0.2, 0.1]))

        # Use factory to construct a similarity function
        cosine_greedy = CosineGreedy(tolerance=0.2)

        score = cosine_greedy.pair(reference, query)

        print(f"Cosine score is {score['score']:.2f} with {score['matches']} matched peaks")

    Should output

    .. testoutput::

        Cosine score is 0.83 with 1 matched peaks

    """
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
            The power to raise m/z to in the cosine function. The default is 0, in which
            case the peak intensity products will not depend on the m/z ratios.
        intensity_power:
            The power to raise intensity to in the cosine function. The default is 1.
        """
        self.tolerance = tolerance
        self.mz_power = mz_power
        self.intensity_power = intensity_power

    def matrix(self, references: List[SpectrumType], queries: List[SpectrumType],
               array_type: str = "numpy",
               is_symmetric: bool = False) -> np.ndarray:
        """Optional: Provide optimized method to calculate an np.array of similarity scores
        for given reference and query spectrums. If no method is added here, the following naive
        implementation (i.e. a double for-loop) is used.

        Parameters
        ----------
        references
            List of reference objects
        queries
            List of query objects
        array_type
            Specify the output array type. Can be "numpy" or "sparse".
            Default is "numpy" and will return a numpy array. "sparse" will return a COO-sparse array.
        is_symmetric
            Set to True when *references* and *queries* are identical (as for instance for an all-vs-all
            comparison). By using the fact that score[i,j] = score[j,i] the calculation will be about
            2x faster.
        """
        #pylint: disable=too-many-locals


            #     if is_symmetric and self.is_commutative:
            #             score = self.pair(reference, query)
            #             if self.keep_score(score):
            #                 idx_row += [i_ref, i_query]
            #                 idx_col += [i_query, i_ref]
            #                 scores += [score, score]
            #     else:
            #         for i_query, query in enumerate(queries[:n_cols]):
            #             score = self.pair(reference, query)
            #             if self.keep_score(score):
            #                 idx_row.append(i_ref)
            #                 idx_col.append(i_query)
            #                 scores.append(score)
            # return scores

        def to_tensor(spectra: List[SpectrumType]) -> np.ndarray:
            max_len = max(len(s.peaks) for s in spectra)
            tensor = np.zeros((len(spectra), max_len, 2), dtype='float32')
            lens = np.zeros((len(spectra)), dtype='int32')
            for i, s in enumerate(spectra):
                peaks = s.peaks.to_numpy
                tensor[i, :len(s.peaks), :] = peaks
                lens[i] = len(peaks)
            return tensor, lens
            
        R, Q = len(references), len(queries)
        scores_array = np.zeros((R, Q), dtype=self.score_datatype)

        rtensor, rlen = to_tensor(references)
        qtensor, qlen = to_tensor(queries)
        lens = np.stack([rlen, qlen], axis=1)

        cosine_greedy_kernel(
            self.tolerance,
            self.mz_power,
            self.intensity_power,

            rtensor,
            qtensor,
            lens,

            scores_array
        )
        # idx_row = np.array(idx_row)
        # idx_col = np.array(idx_col)
        # scores_data = np.array(scores, dtype=self.score_datatype)
        # TODO: make StackedSpareseArray the default and add fixed function to output different formats (with code below)
        if array_type == "numpy":
            # scores_array = np.zeros(shape=(n_rows, n_cols), dtype=self.score_datatype)
            # scores_array[idx_row, idx_col] = scores_data.reshape(-1)
            return scores_array
        if array_type == "sparse":
            scores_array = StackedSparseArray(R, Q)
            scores_array.add_sparse_data(idx_row, idx_col, scores_data, "")
            return scores_array
        raise ValueError("array_type must be 'numpy' or 'sparse'.")

    def pair(self, reference: SpectrumType, query: SpectrumType) -> Tuple[float, int]:
        """Calculate cosine score between two spectra.

        Parameters
        ----------
        reference
            Single reference spectrum.
        query
            Single query spectrum.

        Returns
        -------
        Score
            Tuple with cosine score and number of matched peaks.
        """
        def get_matching_pairs():
            """Get pairs of peaks that match within the given tolerance."""
            matching_pairs = collect_peak_pairs(spec1, spec2, self.tolerance,
                                                shift=0.0, mz_power=self.mz_power,
                                                intensity_power=self.intensity_power)
            if matching_pairs is None:
                return None
            matching_pairs = matching_pairs[np.argsort(matching_pairs[:, 2], kind='mergesort')[::-1], :]
            return matching_pairs

        spec1 = reference.peaks.to_numpy
        spec2 = query.peaks.to_numpy
        matching_pairs = get_matching_pairs()
        if matching_pairs is None:
            return np.asarray((float(0), 0), dtype=self.score_datatype)
        score = score_best_matches(matching_pairs, spec1, spec2,
                                   self.mz_power, self.intensity_power)
        return np.asarray(score, dtype=self.score_datatype)
