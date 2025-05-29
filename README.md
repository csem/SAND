# SAND: One-Shot Feature Selection with Additive Noise Distortion

Code for [SAND Paper](https://arxiv.org/abs/2505.03923) - (ICML 2025).
SAND is a feature selection algorithm for neural networks based on simple addition of a noise layer.

## Experiments
This project was built using `Python 3.9.2`

To download the datasets run `sh sand/experiments/get_all_data.sh`

To run the experiments for `SAND` on the 6 standard datasets, run `python -m sand.experiments.experiment`

**Note:** To run the experiments for `SAND` on the other datasets, change the dataset name and the corresponding number of features to select in the `experiment` file. Additionally, you can change the algorithm to run the experiments for other methods.





## Acknowledgements

This repository is an edited clone of [Sequential Attention](XXXX-1). The original code has been modified to suit the specific needs of this project. All credit for the original code goes to the authors of the original repository.
