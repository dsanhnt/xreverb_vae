# xrverb_vae
Python Source Code of **Extracting the X-Ray Reverberation Response Functions from the Active Galactic Nucleus Light Curves Using an Autoencoder** \
[Deesamutara et. al (2025), ApJ 980:257](https://ui.adsabs.harvard.edu/link_gateway/2025ApJ...980..257D/doi:10.3847/1538-4357/adae85)

**Prerequisits**
- Astropy
- Matplotlib
- Numpy
- Pandas
- PyTorch
- Scipy
- Scikit-Learn
---

Users may use their own simulated lightcurves and responses, for training and deploy the model to a real data. Checkpoint file will be written every 100 iterations. At the end of the computation, predicted response functions and  shall be saves as `.json` format.

---
BibTeX for article

```
@ARTICLE{2025ApJ...980..257D,
       author = {{Deesamutara}, Sanhanat and {Chainakun}, Poemwai and {Worrakitpoonpon}, Tirawut and {Khanthasombat}, Kamonwan and {Luangtip}, Wasutep and {Jiang}, Jiachen and {Pozo Nu{\~n}ez}, Francisco and {Young}, Andrew J.},
        title = "{Extracting the X-Ray Reverberation Response Functions from the Active Galactic Nucleus Light Curves Using an Autoencoder}",
      journal = {\apj},
     keywords = {Reverberation mapping, X-ray astronomy, Active galactic nuclei, Black hole physics, 2019, 1810, 16, 159, High Energy Astrophysical Phenomena, Instrumentation and Methods for Astrophysics},
         year = 2025,
        month = feb,
       volume = {980},
       number = {2},
          eid = {257},
        pages = {257},
          doi = {10.3847/1538-4357/adae85},
archivePrefix = {arXiv},
       eprint = {2501.14618},
 primaryClass = {astro-ph.HE},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025ApJ...980..257D},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

```
