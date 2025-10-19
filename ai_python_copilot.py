# Anscombe Quartet Regression Analysis Python
# ------------------------------------
# This script loads the Anscombe dataset, runs linear regression for each dataset,
# and compares regression coefficients and R-squared values.

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import time

# Loading the Anscombe dataset
anscombe = sns.load_dataset("anscombe")

# Confirm the dataset structure
# print("Anscombe dataset head:\n")
# print(anscombe.head())

# Run regression model
results = []

# Start of time
start = time.time()

for dataset_name in anscombe['dataset'].unique():
    subset = anscombe[anscombe['dataset'] == dataset_name]

    # Perform linear regression: y = slope * x + intercept
    slope, intercept, r_value, p_value, std_err = stats.linregress(subset['x'], subset['y'])

    results.append({
        "Dataset": dataset_name,
        "Intercept": intercept,
        "Slope": slope,
        "R-squared": r_value**2
    })


# end of timer
end = time.time()

# Display regression results and time of execution
df_results = pd.DataFrame(results)
print("\nLinear Regression Results for Anscombe Quartet:\n")
print(df_results)
print(f"Script execution time: {end - start:.6f} seconds")


# Plotting anscombe datasets with regression lines
sns.lmplot(
    x="x", y="y",
    col="dataset",
    hue="dataset",
    data=anscombe,
    col_wrap=2,
    ci=None,
    palette="muted",
    height=4,
    scatter_kws={"s": 50, "alpha": 0.7}
)

plt.suptitle("Anscombe’s Quartet Regression Lines")
plt.show()

