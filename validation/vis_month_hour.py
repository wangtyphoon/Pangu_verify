import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read data
monthly_mean = pd.read_csv("monthly_mean_rmse.csv")
monthly_std = pd.read_csv("monthly_std_rmse.csv")
hourly_mean = pd.read_csv("hourly_mean_rmse.csv")
hourly_std = pd.read_csv("hourly_std_rmse.csv")

# Variables list (excluding month)
variables = ['z500_rmse', 'q700_rmse', 't850_rmse', 'u850_rmse', 'v850_rmse', 'overall_rmse']

# Set font (use default sans-serif for English)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Create monthly variation plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, var in enumerate(variables):
    ax = axes[i]
    
    months = monthly_mean['month']
    mean_vals = monthly_mean[var]
    std_vals = monthly_std[var]
    
    ax.plot(months, mean_vals, 'b-', marker='o', linewidth=2, label='Mean')
    ax.fill_between(months, 
                    mean_vals - std_vals, 
                    mean_vals + std_vals, 
                    alpha=0.3, color='blue', label='Std Dev Range')
    
    ax.set_title(f'{var} - Monthly Variation')
    ax.set_xlabel('Month')
    ax.set_ylabel('RMSE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(months)

plt.tight_layout()
plt.savefig('monthly_rmse_trends.png', dpi=300, bbox_inches='tight')
plt.show()

# Create hourly variation plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, var in enumerate(variables):
    ax = axes[i]
    
    hours = hourly_mean['hour']
    mean_vals = hourly_mean[var]
    std_vals = hourly_std[var]
    
    ax.plot(hours, mean_vals, 'r-', marker='s', linewidth=2, label='Mean')
    ax.fill_between(hours, 
                    mean_vals - std_vals, 
                    mean_vals + std_vals, 
                    alpha=0.3, color='red', label='Std Dev Range')
    
    ax.set_title(f'{var} - Hourly Variation')
    ax.set_xlabel('Hour (UTC)')
    ax.set_ylabel('RMSE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(hours)

plt.tight_layout()
plt.savefig('hourly_rmse_trends.png', dpi=300, bbox_inches='tight')
plt.show()
