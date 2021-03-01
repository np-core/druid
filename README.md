<p align="left"><img src="logo/logo.png" height="115" width="110"></img></p>

# Druid

![](https://img.shields.io/badge/tf--gpu-1.12.0-blue.svg)
![](https://img.shields.io/badge/keras-2.1.6-blue.svg)
![](https://img.shields.io/badge/docs-latest-green.svg)
![](https://img.shields.io/badge/preprint-soon-green.svg)
![](https://img.shields.io/badge/lifecycle-experimental-orange.svg)

 **`v0.1`**`internal release, no tests`

`Druid` is a fork of the hybrid neural net platform for training, evaluating and deploying neural network models that originally act as taxonomic classifiers of raw nanopore signal (`Achilles`). It reimplements the basecaller function of `Chiron` as a Keras model with the specific purpose to detect non canonical base modifications.

<p align="center"><img src="logo/achilles_pretrained.png"></img></p>

### Install
---

`Druid` can be installed with **`Python 3.6`**:

```
pip install git+https://github.com/esteinig/druid  # does not install tensorflow-gpu or keras
```

It requires `PoreMongo`, which can for now be installed with:

```
pip install git+https://github.com/esteinig/poremongo@v0.3-pre
```

You know if the driver and `tensorflow-gpu` work when you call the main help interface of `Achilles`:

```
achilles --help
```

### Documentation
---

Coming soon.

[achilles.readthedocs.io](https://achilles.readthedocs.io)

### Contributors
---

* Eike Steinig
* Lachlan Coin
