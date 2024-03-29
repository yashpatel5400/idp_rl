from idp_rl.molecule_generation import generate_molecule_config


def test_alkanes(mocker):
    cfg1 = generate_molecule_config.branched_alkane(10)
    assert cfg1.mol.GetNumAtoms() == 32
    cfg2 = generate_molecule_config.straight_alkane(12)
    assert cfg2.mol.GetNumAtoms() == 38
    cfg3 = generate_molecule_config.xorgate(2, 4)
    assert cfg3.mol.GetNumAtoms() == 58

def test_custom(mocker):
    chem = mocker.patch('idp_rl.molecule_generation.molecules.Chem')
    chem.AddHs.return_value = 'mol'
    allchem = mocker.patch('idp_rl.molecule_generation.molecules.AllChem')
    normalizers = mocker.patch('idp_rl.molecule_generation.molecules.calculate_normalizers')
    normalizers.return_value = (67, 89)

    cfg1 = generate_molecule_config.test_alkane()
    assert cfg1.tau == 503
    assert cfg1.E0 == 7.668625034772399

    cfg2 = generate_molecule_config.mol_from_molFile('filename')
    chem.MolFromMolFile.assert_called_with('filename')
    assert allchem.MMFFSanitizeMolecule.call_count == 2

    assert cfg2.E0 == 67
    assert cfg2.Z0 == 89

def test_custom2(mocker):
    chem = mocker.patch('idp_rl.molecule_generation.molecules.Chem')
    chem.AddHs.return_value = 'mol'
    allchem = mocker.patch('idp_rl.molecule_generation.molecules.AllChem')
    normalizers = mocker.patch('idp_rl.molecule_generation.molecules.calculate_normalizers')
    normalizers.return_value = (67, 89)


    cfg2 = generate_molecule_config.mol_from_smiles('smiles')
    chem.MolFromSmiles.assert_called_with('smiles')
    allchem.MMFFSanitizeMolecule.assert_called_once()

    assert cfg2.E0 == 67
    assert cfg2.Z0 == 89




