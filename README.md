# prefGP

`prefGP` is a Gaussian process based library for learning from preference and choice data. It is implemented using Jax and PyTorch. 
`prefGP` implements 9 models to learn from preference and choice data:

* Model 1: Consistent Preferences.
* Model 2: Just Noticeable Difference 
* Model 3: Probit for Erroneous Preferences
* Model 4: Preferences with Gaussian noise error
* Model 5: Probit  for Erroneous preferences as a classification problem
* Model 6: Thurstonian model for label preferences
* Model 7: Plackett-Luce model for label ordering data
* Model 8: Paired comparison for label preferences
* Model 9: Rational and Pseudo-rational models for choice data

## Installation

**Requirements**:

* Python == 3.11.4

Download the repository  and then install

```bash
pip install -r requirements.txt
```
## Example
The `notebooks` folder includes several ipython notebooks that demonstrate the use of prefGP. For more details about the models used in the examples, please see the below paper.

## Citing Us
```
@article{prefGP2024,
  title = {A tutorial on learning from preferences and choices with Gaussian Processes},
  author = {Benavoli, Alessio and Azzimonti, Dario},
  journal = {arXiv preprint},
  year = {2024},
  eprint = {2403.11782},
  url = {https://arxiv.org/abs/2403.11782}
}
```
## The Team
The library was developed by
- [Dario Azzimonti](https://sites.google.com/view/darioazzimonti/home) (IDSIA)
- [Alessio Benavoli](https://alessiobenavoli.com/) (Trinity College Dublin)

