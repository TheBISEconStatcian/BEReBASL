# Quantifying the Unknown: A numerical analysis of surprise-based observability of acceptance bias in reject inference

**Type:** Master's Thesis

**Author:** Mateo Ordóñez Serna

**1st Examiner:** Prof. Dr. Stefan Lessmann 

**2nd Examiner:** Dr. Gábor Uhrin 

## Table of Content

- [Summary](#summary)
- [Working with the repo](#Working-with-the-repo)
    - [Dependencies](#Dependencies)
    - [Setup](#Setup)
- [Reproducing results](#Reproducing-results)
    - [Training code](#Training-code)
    - [Evaluation code](#Evaluation-code)
    - [Pretrained models](#Pretrained-models)
- [Results](#Results)
- [Project structure](-Project-structure)

## Summary

  The Reject Inference (RI) problem concerns the challenge of learning from
  outcomes that are only observed for accepted applicants, and of using information
  from rejected applicants whose outcomes remain unknown. This selective observation
  induces acceptance bias and motivates methods for its identification and correction.
  This thesis introduces a dynamic perspective on RI by proposing surprise as a signal
  of acceptance bias. *Surprise* is defined as the discrepancy between the expected
  performance of a classifier, as estimated from historical data, and its realized
  performance over subsequent acceptance rounds. The central hypothesis is that this
  discrepancy contains information about the underlying acceptance bias and therefore
  provides a means of observing it indirectly. First, the RI problem is formulated within
  a measure-theoretic framework, making the underlying assumptions explicit. Second, the
  hypothesis that surprise provides an observable signal of acceptance bias is formally
  formulated. Third, a controlled simulation study investigates the extent to which surprise
  reflects the underlying bias. Thereby the bias could be identified by the proposed framework
  but surprise proved to capture only part of this bias. Finally, possible applications of
  the proposed signal and directions for future research are discussed.

**Keywords**: Dynamic Reject Inference, Surprise, Acceptance Bias, self-reinforcing bias
mechanism

**Full text**: The full text of the thesis can be generated from `.thesis/thesis.qmd`,
the result can also be found under `.thesis/_manuscript/thesis.pdf`

## Working with the repo

### Dependencies

The repository was developed using python 3.13.7

### Setup

[This is an example]

1. Clone this repository

2. Create an virtual environment and activate it
```bash
python -m venv thesis-env
source thesis-env/bin/activate
```

3. Install requirements
```bash
pip install --upgrade pip
pip install -r berebasl/requirements.txt
```

## Reproducing results

The full experiments used in the thesis can be reproduced from the file
`experiments/acceptance_loop_cv_based_MNAR.py/

### Training code

Does a repository contain a way to train/fit the model(s) described in the paper?

### Evaluation code

Does a repository contain a script to calculate the performance of the trained model(s) or run experiments on models?

### Pretrained models

Does a repository provide free access to pretrained model weights?

## Results

Does a repository contain a table/plot of main results and a script to reproduce those results?

## Project structure

(Here is an example from SMART_HOME_N_ENERGY, [Appliance Level Load Prediction](https://github.com/Humboldt-WI/dissertations/tree/main/SMART_HOME_N_ENERGY/Appliance%20Level%20Load%20Prediction) dissertation)

```bash
├── README.md
├── requirements.txt                                -- required libraries
├── data                                            -- stores csv file 
├── plots                                           -- stores image files
└── src
    ├── prepare_source_data.ipynb                   -- preprocesses data
    ├── data_preparation.ipynb                      -- preparing datasets
    ├── model_tuning.ipynb                          -- tuning functions
    └── run_experiment.ipynb                        -- run experiments 
    └── plots                                       -- plotting functions                 
```
