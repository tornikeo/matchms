import numpy as np
import pytest
from matchms.similarity import CosineGreedy
from matchms import Spectrum
from ..builder_Spectrum import SpectrumBuilder
from matchms.importing import load_from_mgf
import os


@pytest.fixture
def pesticides() -> list[Spectrum]:
    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectra_file = os.path.join(module_root, "testdata", "pesticides.mgf")
    spectra = list(load_from_mgf(spectra_file))
    return spectra


@pytest.mark.parametrize('scale', range(1, 5))
def test_cosine_greedy_performance(pesticides, scale, benchmark):
    references = pesticides * scale
    queries = pesticides * scale
    cosine_greedy = CosineGreedy()
    # Warm up (trigger compilation)
    cosine_greedy.matrix(references=references[:4], queries=queries[:4])
    # Benchmark
    benchmark(cosine_greedy.matrix, references=references, queries=queries)