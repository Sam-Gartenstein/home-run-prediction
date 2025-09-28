#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import arviz as az
import matplotlib.pyplot as plt

def plot_trace_summary(trace, var_names=None, tight=True):
    """
    Plot posterior trace plots and densities for selected parameters.

    Parameters
    ----------
    trace : arviz.InferenceData
        Posterior samples from the model.
    var_names : list of str, optional
        Names of variables to include. If None, all are plotted.
    tight : bool, default=True
        Whether to apply tight_layout() for cleaner spacing.
    """
    az.plot_trace(trace, var_names=var_names)
    if tight:
        plt.tight_layout()
    plt.show()


def summarize_posterior(trace, var_names=None, round_to=3, hdi_prob=0.95):
    """
    Generate a summary table of posterior estimates.

    Parameters
    ----------
    trace : arviz.InferenceData
        Posterior samples from the model.
    var_names : list of str, optional
        Variables to include in the summary. If None, all are included.
    round_to : int, default=3
        Number of decimals to round the results to.
    hdi_prob : float, default=0.95
        Probability for the highest density interval (HDI).

    Returns
    -------
    pandas.DataFrame
        Summary statistics including mean, sd, and HDI intervals.
    """
    return az.summary(
        trace,
        var_names=var_names,
        round_to=round_to,
        hdi_prob=hdi_prob
    )

