# bPK score - Prediction of small molecule developability using large-scale in silico ADMET models

This is the code of the bPK score from the paper *Prediction of Small-Molecule Developability Using Large-Scale In Silico ADMET Models* by Beckers et al., *Journal of Medicinal Chemistry*, https://pubs.acs.org/doi/full/10.1021/acs.jmedchem.3c01083


The bPK score is a deep learning based small molecule scoring approach. The model takes as input the predicted ADMET profile of a compounds and is trained on historical pre-clinical development milestones using rank-consistent ordinal regression.

The model is based on PyTorch. Training of the model is described in the Jupyter Notebook train.ipynb