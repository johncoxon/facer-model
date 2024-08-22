# birkeland

A Python implementation of the Birkeland current model presented in [Milan, S. E. (2013), Modeling Birkeland currents in the expanding/contracting polar cap paradigm, J. Geophys. Res. Space Physics, 118, 5532– 5542, doi:10.1002/jgra.50393](https://doi.org/10.1002/jgra.50393).

A simple mathematical model of the region 1 and 2 Birkeland current system intensities for differing dayside and nightside magnetic reconnection rates, consistent with the expanding/contracting polar cap paradigm of solar wind-magnetosphere-ionosphere coupling.

## Tests

```
pytest tests/test.py
```

## Usage

```
dayside_reconnection_rate = 50      # in kV
nightside_reconnection_rate = 50    # in kV
polar_cap_flux = 0.5                # in GWb

model = Birkeland(dayside_reconnection_rate, nightside_reconnection_rate, polar_cap_flux)
```