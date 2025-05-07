import pytest
import numpy as np
from montecarlo import BitString  # Replace with your actual module import

# Test Initialization
def test_initialization():
    bitstring = BitString(5)
    assert len(bitstring) == 5
    assert np.array_equal(bitstring.config, np.zeros(5, dtype=int))

# Test __repr__
def test_repr():
    bitstring = BitString(5)
    assert repr(bitstring) == "00000"  # Initially all bits are 0
    bitstring.config = np.array([1, 0, 1, 1, 0])
    assert repr(bitstring) == "10110"

# Test __eq__
def test_eq():
    bitstring1 = BitString(5)
    bitstring2 = BitString(5)
    bitstring1.config = np.array([1, 0, 1, 1, 0])
    bitstring2.config = np.array([1, 0, 1, 1, 0])
    assert bitstring1 == bitstring2  # Same config, should be equal
    
    bitstring2.config = np.array([0, 1, 1, 1, 0])
    assert bitstring1 != bitstring2  # Different config, should not be equal

# Test on() and off()
def test_on_and_off():
    bitstring = BitString(5)
    assert bitstring.on() == 0
    assert bitstring.off() == 5

    bitstring.config = np.array([1, 1, 0, 1, 0])
    assert bitstring.on() == 3
    assert bitstring.off() == 2

# Test flip_site()
def test_flip_site():
    bitstring = BitString(5)
    bitstring.flip_site(2)
    assert bitstring.config[2] == 1

    with pytest.raises(ValueError):
        bitstring.flip_site(6)  # Out of bounds

# Test integer()
def test_integer():
    bitstring = BitString(5)
    bitstring.config = np.array([1, 0, 1, 1, 0])
    assert bitstring.integer() == 22  # 10110 in binary is 22 in decimal

# Test set_config()
def test_set_config():
    bitstring = BitString(5)
    bitstring.set_config([1, 0, 1, 0, 1])
    assert np.array_equal(bitstring.config, np.array([1, 0, 1, 0, 1]))

    with pytest.raises(ValueError):
        bitstring.set_config([1, 0])  # Invalid config length

# Test set_integer_config()
def test_set_integer_config():
    bitstring = BitString(5)
    bitstring.set_integer_config(22)  # 10110 in binary
    assert np.array_equal(bitstring.config, np.array([1, 0, 1, 1, 0]))

    with pytest.raises(ValueError):
        bitstring.set_integer_config(32)  # Decimal value exceeds the range for length 5

# Test copy()
def test_copy():
    bitstring = BitString(5)
    bitstring.config = np.array([1, 0, 1, 0, 1])
    bitstring_copy = bitstring.copy()
    assert bitstring == bitstring_copy  # Same config
    assert bitstring is not bitstring_copy  # Different objects
