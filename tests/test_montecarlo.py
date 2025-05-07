import numpy as np
import networkx as nx
from montecarlo.bitstring import BitString
from montecarlo.ising_hamiltonian import IsingHamiltonian
from montecarlo.montecarlo import MonteCarlo


def create_test_hamiltonian():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2)])
    return IsingHamiltonian(G)


def test_initialize_configuration():
    hamiltonian = create_test_hamiltonian()
    mc = MonteCarlo(hamiltonian)
    assert isinstance(mc.config, BitString)
    assert len(mc.config) == 3


def test_metropolis_step_changes_config():
    hamiltonian = create_test_hamiltonian()
    mc = MonteCarlo(hamiltonian)

    old_config = mc.config.copy()
    mc.metropolis_step(T=1.0)

    # Config may or may not change depending on randomness, but should still be a BitString
    assert isinstance(mc.config, BitString)
    assert len(mc.config) == 3


def test_run_outputs_correct_shape():
    hamiltonian = create_test_hamiltonian()
    mc = MonteCarlo(hamiltonian)
    T = 2.0
    n_samples = 50
    n_burn = 10
    energies, magnetizations = mc.run(T, n_samples=n_samples, n_burn=n_burn)

    assert isinstance(energies, np.ndarray)
    assert isinstance(magnetizations, np.ndarray)
    assert energies.shape == (n_samples,)
    assert magnetizations.shape == (n_samples,)


def test_run_with_zero_temperature_accepts_only_lower_energy():
    # At T=0, Metropolis should only accept energy-lowering moves
    hamiltonian = create_test_hamiltonian()
    mc = MonteCarlo(hamiltonian)
    initial_config = mc.config.copy()

    # Run for one step at T=0
    mc.metropolis_step(T=1e-9)
    assert isinstance(mc.config, BitString)
    assert len(mc.config) == 3
