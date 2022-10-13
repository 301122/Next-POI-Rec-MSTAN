# Next-POI-Rec-MSTAN

[![DOI](https://zenodo.org/badge/546544595.svg)](https://zenodo.org/badge/latestdoi/546544595)

## Environment

The codes of MSTAN are implemented and tested under the following development environment:

- python=3.7.13
- tensorflow-gpu=2.5.0
- numpy=1.19.5

## Datasets

We used two datasets, `Foursquare` and `Gowalla`, as mentioned in our paper.

The datasets are divided into training set, validation set and testing set by 8:1:1.

## How to Run the Codes

We provided the **Foursquare** dataset in `data/Foursquare`. 

You can run the code from the following command line:

```
python main.py
```

In addition, you can debug hyperparameters in code.
