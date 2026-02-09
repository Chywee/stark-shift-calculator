# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Commands

```bash
# Run with example calculations
python stark_shift.py

# Import as module
python -c "from stark_shift import quadratic_stark_shift; print(quadratic_stark_shift(n=2, electric_field=1e6))"
```

## Dependencies

- Python 3.7+
- NumPy

Install: `pip install numpy`

## Architecture

Single-module physics calculator (`stark_shift.py`) implementing perturbation theory results for the Stark effect in hydrogen:

- `quadratic_stark_shift()` - 2nd order perturbation (all states)
- `linear_stark_shift()` - 1st order perturbation (n ≥ 2 only, uses parabolic quantum number k)
- `total_stark_shift()` - combines both contributions
- `print_stark_shift()` - formatted console output

## Conventions

- All functions return results in multiple units: Joules, eV, Hz, cm⁻¹
- Electric field input is always in SI units (V/m)
- Internal calculations use atomic units (Hartree, Bohr radius) then convert to SI
- Physical constants are defined at module level (lines 10-14)
