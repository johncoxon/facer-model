# facer-model

[![PyPI badge](https://badge.fury.io/py/facer-model.svg)](https://badge.fury.io/py/facer-model)
[![Zenodo badge](https://zenodo.org/badge/665918230.svg)](https://doi.org/10.5281/zenodo.15731338)
[![Tests badge](https://github.com/johncoxon/birkeland-model/actions/workflows/tests.yaml/badge.svg)](https://github.com/johncoxon/birkeland-model/actions/workflows/tests.yaml)

Field-Aligned Currents Estimated from Reconnection (FACER) is a Python implementation of the Birkeland current model presented by Coxon et al. (2016), based on Milan (2013). It is a simple mathematical model of the region 1 and 2 Birkeland current system intensities for differing dayside and nightside magnetic reconnection rates, consistent with the expanding/contracting polar cap paradigm of solar wind-magnetosphere-ionosphere coupling.

## Citations

When using this software, please cite [the Zenodo record](https://doi.org/10.5281/zenodo.15731338) as well as the following two papers:

- [Stephen E Milan (2013), Modeling Birkeland currents in the expanding/contracting polar cap paradigm, _Journal of Geophysical Research: Space Physics_, 118, 5532–5542. https://doi.org/10.1002/jgra.50393](https://doi.org/10.1002/jgra.50393)
- [John C Coxon, Stephen E Milan, Jennifer A Carter, Lasse BN Clausen, Brian J Anderson & Haje Korth (2016). Seasonal and diurnal variations in AMPERE observations of the Birkeland currents compared to modeled results, _Journal of Geophysical Research: Space Physics_, 121, 4027–4040. https://doi.org/10.1002/2015JA022050](https://doi.org/10.1002/2015JA022050)

## Funding

John C Coxon was supported during this work by Science and Technology Facilities Council (STFC) Ernest Rutherford Fellowship ST/V004883/1.

## Installation

To install and test the model, simply type:

```
pip install facer-model
pytest tests/test.py
```

## Usage

A very basic instantiation of the model can be seen below. For more examples, check out the notebooks in the repository, which explicitly detail how to use the model to replicate the results from the two papers which define the model.
```
from facer import Model

dayside_reconnection_rate = 50       # in kV
nightside_reconnection_rate = 50     # in kV
solar_radio_flux = 80                # in SFU
time = datetime.datetime(2010, 1, 1)
hemisphere = "north"
polar_cap_flux = 0.5                 # in GWb

model = Model(dayside_reconnection_rate,
              nightside_reconnection_rate,
              solar_radio_flux,
              time,
              hemisphere,
              polar_cap_flux)
```
