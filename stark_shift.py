"""
Stark Shift Calculator for Hydrogen-like Atoms

Calculates the Stark shift (energy shift due to external electric field)
for a given principal quantum number n and electric field strength.
"""

import numpy as np
import matplotlib.pyplot as plt

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


def plot_stark_levels(n: int, electric_field: float, m: int = 0, unit: str = 'ev'):
    """
    Plot energy level diagram showing unperturbed and Stark-shifted levels.
    
    Parameters
    ----------
    n : int
        Principal quantum number (n >= 2 for interesting splitting)
    electric_field : float
        Electric field strength in V/m
    m : int, optional
        Magnetic quantum number, default is 0
    unit : str, optional
        Unit for energy display: 'ev', 'hz', 'cm_inv', or 'joules'
    """
    # Unperturbed energy: E_n = -13.6 eV / n²
    E_unperturbed_ev = -13.6 / n**2
    
    # Convert to requested unit
    unit_labels = {
        'ev': ('eV', 1),
        'hz': ('Hz', EV_TO_JOULE / 6.62607015e-34),
        'cm_inv': ('cm⁻¹', EV_TO_JOULE / 6.62607015e-34 / 2.99792458e10),
        'joules': ('J', EV_TO_JOULE)
    }
    
    if unit not in unit_labels:
        raise ValueError(f"Unit must be one of {list(unit_labels.keys())}")
    
    unit_label, conversion = unit_labels[unit]
    E_unperturbed = E_unperturbed_ev * conversion
    
    # Calculate all possible k values: -(n-1-|m|) to (n-1-|m|)
    k_max = n - 1 - abs(m)
    k_values = list(range(-k_max, k_max + 1))
    
    # Calculate shifted energies for each k
    shifted_energies = []
    labels = []
    
    for k in k_values:
        shift = total_stark_shift(n, k, electric_field, m)
        shift_value = shift[f'shift_{unit}'] if unit != 'ev' else shift['shift_ev']
        if unit == 'ev':
            E_shifted = E_unperturbed_ev + shift_value
        else:
            E_shifted = E_unperturbed + shift_value * conversion if unit == 'ev' else E_unperturbed + shift[f'shift_{unit}']
        
        # Recalculate properly
        E_shifted = E_unperturbed + shift[f'shift_{unit}'] if unit != 'ev' else E_unperturbed_ev + shift['shift_ev']
        shifted_energies.append(E_shifted)
        labels.append(f'k={k}')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot unperturbed level (left side)
    ax.hlines(E_unperturbed if unit != 'ev' else E_unperturbed_ev, 0, 0.4, 
              colors='blue', linewidth=3, label=f'Non perturbé (n={n})')
    ax.text(0.2, (E_unperturbed if unit != 'ev' else E_unperturbed_ev), 
            f'n={n}', ha='center', va='bottom', fontsize=12, color='blue')
    
    # Plot shifted levels (right side)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(k_values)))
    
    for i, (E, k, label) in enumerate(zip(shifted_energies, k_values, labels)):
        ax.hlines(E, 0.6, 1.0, colors=colors[i], linewidth=2)
        ax.text(1.02, E, label, ha='left', va='center', fontsize=9, color=colors[i])
    
    # Draw connecting lines
    E_ref = E_unperturbed if unit != 'ev' else E_unperturbed_ev
    for i, E in enumerate(shifted_energies):
        ax.plot([0.4, 0.6], [E_ref, E], 'k--', alpha=0.3, linewidth=0.8)
    
    # Formatting
    ax.set_xlim(-0.1, 1.3)
    
    # Set y-axis limits with some padding
    all_energies = [E_ref] + shifted_energies
    y_min, y_max = min(all_energies), max(all_energies)
    padding = (y_max - y_min) * 0.1 if y_max != y_min else abs(y_min) * 0.1
    ax.set_ylim(y_min - padding, y_max + padding)
    
    ax.set_ylabel(f'Énergie ({unit_label})', fontsize=12)
    ax.set_title(f'Effet Stark pour H (n={n}, F={electric_field:.2e} V/m, m={m})', fontsize=14)
    
    # Remove x-axis ticks and add labels
    ax.set_xticks([0.2, 0.8])
    ax.set_xticklabels(['Sans champ', 'Avec champ\n(effet Stark)'])
    
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax


if __name__ == "__main__":
    # Example usage
    print("Example: Stark shift for a ryberg atoms")
    
    # Rydberg state
    print_stark_shift(n=150, electric_field=2, m=1, k=1)
    
    # Plot energy level diagram
    plot_stark_levels(n=150, electric_field=2, m=, unit='hz')
    