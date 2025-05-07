import numpy as np
import networkx as nx
import pytest
from montecarlo.bitstring import BitString
from montecarlo.ising_hamiltonian import IsingHamiltonian


def test_energy_no_field():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2)], weight=1)
    ising = IsingHamiltonian(G)

    config = BitString(3)
    config.set_config([1, 1, 1])  # All spins up (1)

    energy = ising.energy(config)
    # All aligned: (1*1 + 1*1) = 2
    assert energy == 2

    config.set_config([1, 0, 1])  # Spin 1 down
    energy = ising.energy(config)
    # Spins: (1*-1 + -1*1) = -2
    assert energy == -2


def test_energy_with_field():
    G = nx.Graph()
    G.add_edges_from([(0, 1)])
    ising = IsingHamiltonian(G)
    ising.set_mu(np.array([1, -1]))

    config = BitString(2)
    config.set_config([1, 0])  # spin config: [1, -1]
    # interaction: 1 * -1 = -1
    # field term: 1*(1) + -1*(-1) = 2
    expected_energy = -1 + 2
    assert ising.energy(config) == expected_energy


def test_set_mu_invalid_length():
    G = nx.Graph()
    G.add_nodes_from([0, 1])
    ising = IsingHamiltonian(G)
    with pytest.raises(ValueError):
        ising.set_mu(np.array([1]))  # wrong length


def test_compute_average_values_basic():
    G = nx.Graph()
    G.add_edges_from([(0, 1)])
    ising = IsingHamiltonian(G)
    T = 1.0

    avg_energy, avg_mag, heat_cap, mag_susc = ising.compute_average_values(T)
    assert isinstance(avg_energy, float)
    assert isinstance(avg_mag, float)
    assert isinstance(heat_cap, float)
    assert isinstance(mag_susc, float)


def test_compute_average_values_invalid_temperature():
    G = nx.Graph()
    G.add_nodes_from([0, 1])
    ising = IsingHamiltonian(G)

    with pytest.raises(ValueError):
        ising.compute_average_values(0)
