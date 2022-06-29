from idp_rl.molecule_generation.generate_chignolin import generate_chignolin

# additional testing done in jupyter notebook
def test_chignolin(mocker):
    mol = generate_chignolin()
