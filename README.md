# ⚾ Home Run Prediction

This project estimates the probability of a home run occurring during a given MLB plate appearance using a **Bayesian hierarchical model**. The model incorporates batter-specific and pitcher-specific effects to make matchup-level predictions and is developed using **PyMC** and **ArviZ**. This notebook is designed to run in **Google Colab**.

As of now, this is a simple model incorporating only the batter and pitcher involved in each plate appearance. Each at-bat is modeled as a Bernoulli trial, with the probability of a home run determined by a latent power parameter for the batter and a suppression parameter for the pitcher. Future versions will incorporate additional covariates such as pitch type, ballpark, and weather conditions to improve predictive accuracy.
