## Label-aggregation

This repo implements a basic Bayesian model for aggregating labels from multiple annotators in a way that takes annotator biases (and optionally vigilance) into account. The method is described in more detail in the appendix of "Detecting Stance in Media on Global Warming" by Yiwei Luo, Dallas Card, and Dan Jurafsky [https://www.aclweb.org/anthology/2020.findings-emnlp.296](https://www.aclweb.org/anthology/2020.findings-emnlp.296)

### Requirements

The updated version of this code uses `numpy`, `scipy` and `pystan=3.7`. To install pystan, follow the instructions on this page: [https://pystan2.readthedocs.io/en/latest/getting_started.html](https://pystan2.readthedocs.io/en/latest/getting_started.html)

### Usage

An example dataset has been provided in `data/example.jsonlist`
To run the script, use:
`python run_pystan3.py python run_pystan3.py data/example.jsonlist data/`
This will run the script and output a number of files to `data/`, the most important of which is `item_probs.json`, which contains the estimated label probabilities for each item.
