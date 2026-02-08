# ⚾️ Home Run Prediction

This project estimates the probability of a home run occurring during a given MLB plate appearance using a **Bayesian hierarchical model**. The model incorporates batter-specific and pitcher-specific effects to make matchup-level predictions and is developed using **PyMC** and **ArviZ**. 

## Partial Pooling & Hierarchical Structure

A key strength of this model is its use of **partial pooling** through a hierarchical Bayesian framework. Rather than estimating completely independent effects for each batter, pitcher, and ballpark, the model allows these groups to **share statistical strength**.

This approach stabilizes estimates for players or parks with limited data by shrinking their effects toward the overall league average, while still allowing well-observed groups to deviate meaningfully. As a result, the model avoids overfitting sparse matchups while preserving genuine signal where data is abundant.

Because information is shared across groups, the model can make reasonable predictions even when direct historical data for a specific matchup is scarce. For example, if a pitcher has faced a batter only a handful of times, the model can still draw on:

- how that pitcher has performed against **similar batters**, and  
- how that batter has performed against **similar pitchers**,  

while also accounting for the **run environment of the ballpark**.

This cross-group information flow improves generalization and produces more realistic predictions in rare or novel situations.

## Coming Soon!

- Workflow
