
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test


df=pd.read_csv('models\your_data2.csv')

# Clean the data
data = df[['fu_time', 'death', 'copd']].dropna()

print(f"Data summary:")
print(f"Total patients: {len(data)}")
print(f"COPD patients: {data['copd'].sum()}")
print(f"Deaths: {data['death'].sum()}")

# Fit Cox model
cph = CoxPHFitter()
cph.fit(data, duration_col='fu_time', event_col='death')

print("\nCox model coefficients:")
print(cph.summary)

# TEST 1: Using Schoenfeld residuals test (the standard method)
print("\n" + "="*60)
print("TEST 1: SCHOENFELD RESIDUALS TEST")
print("="*60)

# This is the standard test for proportional hazards
proportional_hazard_test_results = proportional_hazard_test(cph, data, time_transform='rank')
print(proportional_hazard_test_results.summary)

# Extract p-value
p_value_schoenfeld = proportional_hazard_test_results.summary['p'].iloc[0]
print(f"\nSchoenfeld test p-value: {p_value_schoenfeld:.4f}")

# TEST 2: Using check_assumptions method
print("\n" + "="*60)
print("TEST 2: CHECK_ASSUMPTIONS METHOD")
print("="*60)

try:
    assumptions_results = cph.check_assumptions(data, show_plots=False)
    # If it returns a list, try to extract p-value
    if assumptions_results and len(assumptions_results) > 0:
        if hasattr(assumptions_results[0], 'p_value'):
            p_value_check = assumptions_results[0].p_value
            print(f"Check assumptions p-value: {p_value_check:.4f}")
except Exception as e:
    print(f"Check assumptions method failed: {e}")

# TEST 3: Manual calculation using log-log plots approach
print("\n" + "="*60)
print("TEST 3: LOG-LOG SURVIVAL PLOTS (Visual Check)")
print("="*60)

from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# Create log-log survival plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Regular survival plot
kmf_copd1 = KaplanMeierFitter()
kmf_nocopd1 = KaplanMeierFitter()

kmf_copd1.fit(durations=data[data['copd']==1]['fu_time'], 
              event_observed=data[data['copd']==1]['death'], 
              label='With COPD')
kmf_nocopd1.fit(durations=data[data['copd']==0]['fu_time'], 
                event_observed=data[data['copd']==0]['death'], 
                label='Without COPD')

kmf_copd1.plot_survival_function(ax=ax1, ci_show=False)
kmf_nocopd1.plot_survival_function(ax=ax1, ci_show=False)
ax1.set_title('Survival Curves')
ax1.set_ylabel('Survival Probability')

# Log-log survival plot
kmf_copd2 = KaplanMeierFitter()
kmf_nocopd2 = KaplanMeierFitter()

kmf_copd2.fit(durations=data[data['copd']==1]['fu_time'], 
              event_observed=data[data['copd']==1]['death'], 
              label='With COPD')
kmf_nocopd2.fit(durations=data[data['copd']==0]['fu_time'], 
                event_observed=data[data['copd']==0]['death'], 
                label='Without COPD')

kmf_copd2.plot_survival_function(ax=ax2, ci_show=False)
kmf_nocopd2.plot_survival_function(ax=ax2, ci_show=False)
ax2.set_yscale('log')
ax2.set_ylabel('Log Survival Probability')
ax2.set_title('Log-Log Survival Curves\n(Check for parallel lines)')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"Schoenfeld residuals test p-value: {p_value_schoenfeld:.4f}")
print(f"Rounded to 2 decimal places: {p_value_schoenfeld:.2f}")

if p_value_schoenfeld > 0.05:
    print("✓ Proportional hazards assumption HOLDS for COPD")
else:
    print("✗ Proportional hazards assumption VIOLATED for COPD")