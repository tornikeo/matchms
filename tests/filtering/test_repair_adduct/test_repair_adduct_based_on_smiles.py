import pytest
from matchms.filtering.repair_adduct.repair_adduct_based_on_smiles import repair_adduct_based_on_smiles
from tests.builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("precursor_mz, expected_adduct, ionmode",
                         [(17.0, "[M+H]+", "positive"),
                          (17.5, "[M+H+NH4]2+", "positive"),
                          (74.0, "[2M+ACN+H]+", "positive"),
                          (15.0, "[M-H]-", "negative"),
                          (51.0, "[M+Cl]-", "negative"),
                          (4.33333, "[M-3H]3-", "negative"),
                          ])
def test_repair_adduct_based_on_smiles_not_mol_wt(precursor_mz, expected_adduct, ionmode):
    # CH4 is used as smiles, this has a mass of 16
    spectrum_in = SpectrumBuilder().with_metadata({"smiles": "C", "precursor_mz": precursor_mz, "ionmode": ionmode}).build()
    spectrum_out = repair_adduct_based_on_smiles(spectrum_in, mass_tolerance=0.1, accept_parent_mass_is_mol_wt=False)
    assert spectrum_out.get("adduct") == expected_adduct

