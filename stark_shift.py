"""
Stark Shift Calculator for Hydrogen-like Atoms

Calculates the Stark shift (energy shift due to external electric field)
for a given principal quantum number n and electric field strength.
"""

import numpy as np

# Physical constants (SI units)
E_HARTREE = 4.3597447222071e-18  # Hartree energy in Joules
A_BOHR = 5.29177210903e-11       # Bohr radius in meters
E_CHARGE = 1.602176634e-19       # Elementary charge in Coulombs
EV_TO_JOULE = 1.602176634e-19    # eV to Joules conversion


def quadratic_stark_shift(n: int, electric_field: float, m: int = 0) -> dict:
    """
    Calculate the quadratic Stark shift for hydrogen atom.
    
    The quadratic Stark shift dominates for states where there is no
    linear Stark effect (non-degenerate states) or at low field strengths.
    
    Parameters
    ----------
    n : int
        Principal quantum number (n >= 1)
    electric_field : float
        Electric field strength in V/m
    m : int, optional
        Magnetic quantum number (|m| <= n-1), default is 0
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'shift_joules': Energy shift in Joules
        - 'shift_ev': Energy shift in electron volts
        - 'shift_hz': Energy shift as frequency in Hz
        - 'shift_cm_inv': Energy shift in wavenumbers (cm^-1)
    """
    if n < 1:
        raise ValueError("Principal quantum number n must be >= 1")
    if abs(m) > n - 1:
        raise ValueError(f"|m| must be <= n-1, got m={m} for n={n}")
    
    # Atomic units: F_au = F * (e * a_0 / E_h)
    F_au = electric_field * E_CHARGE * A_BOHR / E_HARTREE
    
    # Quadratic Stark shift formula for hydrogen (in atomic units):
    # ΔE = -(1/16) * F² * n⁴ * (17n² - 3m² - 9|m| + 19)
    # This is the second-order perturbation theory result
    
    shift_au = -(1/16) * F_au**2 * n**4 * (17 * n**2 - 3 * m**2 - 9 * abs(m) + 19)
    
    # Convert to SI units
    shift_joules = shift_au * E_HARTREE
    shift_ev = shift_joules / EV_TO_JOULE
    shift_hz = shift_joules / (6.62607015e-34)  # Planck constant
    shift_cm_inv = shift_hz / (2.99792458e10)   # Speed of light in cm/s
    
    return {
        'shift_joules': shift_joules,
        'shift_ev': shift_ev,
        'shift_hz': shift_hz,
        'shift_cm_inv': shift_cm_inv
    }


def linear_stark_shift(n: int, k: int, electric_field: float) -> dict:
    """
    Calculate the linear Stark shift for hydrogen atom.
    
    The linear Stark effect occurs in hydrogen due to the degeneracy
    of states with different l values for a given n.
    
    Parameters
    ----------
    n : int
        Principal quantum number (n >= 2)
    k : int
        Parabolic quantum number k = n1 - n2 where n1, n2 are
        parabolic quantum numbers. Range: -(n-1) <= k <= (n-1)
    electric_field : float
        Electric field strength in V/m
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'shift_joules': Energy shift in Joules
        - 'shift_ev': Energy shift in electron volts
        - 'shift_hz': Energy shift as frequency in Hz
        - 'shift_cm_inv': Energy shift in wavenumbers (cm^-1)
    """
    if n < 2:
        raise ValueError("Linear Stark effect requires n >= 2")
    if abs(k) > n - 1:
        raise ValueError(f"|k| must be <= n-1, got k={k} for n={n}")
    
    # Atomic units: F_au = F * (e * a_0 / E_h)
    F_au = electric_field * E_CHARGE * A_BOHR / E_HARTREE
    
    # Linear Stark shift formula (first-order perturbation theory):
    # ΔE = (3/2) * n * k * F  (in atomic units)
    
    shift_au = (3/2) * n * k * F_au
    
    # Convert to SI units
    shift_joules = shift_au * E_HARTREE
    shift_ev = shift_joules / EV_TO_JOULE
    shift_hz = shift_joules / (6.62607015e-34)
    shift_cm_inv = shift_hz / (2.99792458e10)
    
    return {
        'shift_joules': shift_joules,
        'shift_ev': shift_ev,
        'shift_hz': shift_hz,
        'shift_cm_inv': shift_cm_inv
    }


def total_stark_shift(n: int, k: int, electric_field: float, m: int = 0) -> dict:
    """
    Calculate the total Stark shift including both linear and quadratic terms.
    
    Parameters
    ----------
    n : int
        Principal quantum number (n >= 1)
    k : int
        Parabolic quantum number k = n1 - n2 (for linear term)
        For n=1 or when k=0, only quadratic shift contributes
    electric_field : float
        Electric field strength in V/m
    m : int, optional
        Magnetic quantum number, default is 0
        
    Returns
    -------
    dict
        Dictionary containing shift values in various units
    """
    result = {'shift_joules': 0, 'shift_ev': 0, 'shift_hz': 0, 'shift_cm_inv': 0}
    
    # Add quadratic contribution
    quad = quadratic_stark_shift(n, electric_field, m)
    for key in result:
        result[key] += quad[key]
    
    # Add linear contribution if applicable
    if n >= 2 and k != 0:
        lin = linear_stark_shift(n, k, electric_field)
        for key in result:
            result[key] += lin[key]
    
    return result


def print_stark_shift(n: int, electric_field: float, m: int = 0, k: int = 0):
    """
    Print the Stark shift in a formatted way.
    
    Parameters
    ----------
    n : int
        Principal quantum number
    electric_field : float
        Electric field strength in V/m
    m : int, optional
        Magnetic quantum number
    k : int, optional
        Parabolic quantum number for linear Stark effect
    """
    print(f"\n{'='*60}")
    print(f"Stark Shift Calculator")
    print(f"{'='*60}")
    print(f"Input parameters:")
    print(f"  Principal quantum number n = {n}")
    print(f"  Electric field F = {electric_field:.3e} V/m")
    print(f"  Magnetic quantum number m = {m}")
    print(f"  Parabolic quantum number k = {k}")
    print(f"{'='*60}")
    
    # Quadratic shift
    quad = quadratic_stark_shift(n, electric_field, m)
    print(f"\nQuadratic Stark shift (2nd order):")
    print(f"  ΔE = {quad['shift_ev']:.6e} eV")
    print(f"  ΔE = {quad['shift_hz']:.6e} Hz")
    print(f"  ΔE = {quad['shift_cm_inv']:.6e} cm⁻¹")
    
    # Linear shift (if applicable)
    if n >= 2:
        lin = linear_stark_shift(n, k, electric_field)
        print(f"\nLinear Stark shift (1st order, k={k}):")
        print(f"  ΔE = {lin['shift_ev']:.6e} eV")
        print(f"  ΔE = {lin['shift_hz']:.6e} Hz")
        print(f"  ΔE = {lin['shift_cm_inv']:.6e} cm⁻¹")
    
    # Total shift
    total = total_stark_shift(n, k, electric_field, m)
    print(f"\nTotal Stark shift:")
    print(f"  ΔE = {total['shift_ev']:.6e} eV")
    print(f"  ΔE = {total['shift_hz']:.6e} Hz")
    print(f"  ΔE = {total['shift_cm_inv']:.6e} cm⁻¹")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Example usage
    print("Example: Stark shift for hydrogen")
    
    # Rydberg state (n=100) 
    print_stark_shift(n=150, electric_field=2, m = 1 ,k  = 1)  
    
    