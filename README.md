# Druid

![](https://img.shields.io/badge/tf--gpu-1.12.0-blue.svg)
![](https://img.shields.io/badge/keras-2.1.6-blue.svg)
![](https://img.shields.io/badge/docs-latest-green.svg)
![](https://img.shields.io/badge/preprint-soon-green.svg)
![](https://img.shields.io/badge/lifecycle-experimental-orange.svg)

 **`v0.1`**`internal release, no tests`

`Druid` is a fork of the hybrid neural net platform for training, evaluating and deploying neural network models that originally act as taxonomic classifiers of raw nanopore signal (`Achilles`). It reimplements the basecaller function of `Chiron` as a Keras model with the specific purpose to detect non canonical base modifications.

### Install
---

```
pip install git+https://github.com/esteinig/druid  # does not install tensorflow-gpu or keras
```

`PoreMongo` dependency, which can for now be installed with:

```
pip install git+https://github.com/esteinig/poremongo
```

### Documentation
---

Coming soon.

[druid.readthedocs.io](https://druid.readthedocs.io)

### Contributors
---

* Eike Steinig
* Lachlan Coin
