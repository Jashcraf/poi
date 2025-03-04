# poi
Extensions to prysm and dygdug for wavefront sensing, coronagraph design, and post-processing. Poi is in a very early development phase and API-breaking changes may occur. 

## Features
- Implicit Electric Field Conjugation
- 4-step Speckle Nulling
- Speckle Area Nulling
- Parametric focus-diverse phase retrieval using algorithmic differentiation with nonlinear optimization
- Apodized Pupil Lyot Coronagraph Design

## Dependecies
Poi's only dependency is on the `prysm` optical propagation package by Brandon Dube, which is in turn dependent on numpy and scipy. However, prysm must be built from source to make use of the latest features that are not in its latest v0.21.1 release. **This must be done before installing Poi**

```bash
git clone https://github.com/Jashcraf/prysm/tree/master
cd prysm
pip install -e .
```

## Installation
Poi can be installed by cloning from source, which will install numpy, scipy, and tqdm for progress bars :D

```bash
git clone https://github.com/Jashcraf/poi
cd poi
pip install -e .
```

## Contributions / Questions
If you wish to contribute to Poi, or have any questions about its use, please open an issue to start a discussion. Before a pull request is made, we prefer that an issue is made to discuss the contributions at a high level.

## Acknowledgements
Enormous gratitude to Brandon Dube for teaching me the basics of algorithmic differentiation (and writing `prysm`, which this is based on). Thanks to Kevin Derby and Kian Milani for testing components of this codebase and demonstrating some of the phase retrieval algorithms.  
