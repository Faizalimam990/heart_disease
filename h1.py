# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif, SelectKBest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load your dataset (adjust the path as necessary)
dataset_path = 'dataset.xlsx'
df = pd.read_excel(dataset_path)

# Clean and preprocess the dataset (removing unnamed columns and handling missing values)
df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col])
df.fillna(df.mean(), inplace=True)

# Separate the features and target variable
X = df.drop(columns='cardio')  # Assuming 'cardio' is the target column
y = df['cardio']

# Apply ANOVA F-Test
selector_anova = SelectKBest(score_func=f_classif, k='all')  # Select all features for scoring
X_anova = selector_anova.fit_transform(X, y)
anova_scores = selector_anova.scores_

# Apply Chi-Square Test (requires non-negative values, so scale features between 0 and 1)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
selector_chi2 = SelectKBest(score_func=chi2, k='all')
X_chi2 = selector_chi2.fit_transform(X_scaled, y)
chi2_scores = selector_chi2.scores_

# Apply Mutual Information
selector_mi = SelectKBest(score_func=mutual_info_classif, k='all')
X_mi = selector_mi.fit_transform(X_scaled, y)
mi_scores = selector_mi.scores_

# Apply PCA (Principal Component Analysis) and compute explained variance
scaler_pca = StandardScaler()
X_pca_scaled = scaler_pca.fit_transform(X)
pca = PCA()
pca.fit(X_pca_scaled)
pca_variance = pca.explained_variance_ratio_

# Create DataFrame for all computed feature scores
df_feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'ANOVA': anova_scores,
    'Chi-Square': chi2_scores,
    'Mutual Information': mi_scores,
    'PCA (Explained Variance)': np.concatenate([pca_variance, np.zeros(len(X.columns) - len(pca_variance))])  # Ensure consistent length
})

# Normalize the feature scores (optional but recommended for better comparison)
df_normalized = df_feature_scores.copy()
score_columns = ['ANOVA', 'Chi-Square', 'Mutual Information', 'PCA (Explained Variance)']
df_normalized[score_columns] = df_normalized[score_columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Calculate the average score across all methods
df_normalized['Average Score'] = df_normalized[score_columns].mean(axis=1)

# Rank the features based on the average score
df_normalized['Rank'] = df_normalized['Average Score'].rank(ascending=False)

# Sort the DataFrame by rank
df_sorted = df_normalized.sort_values(by='Rank')

# Display the ranked features
print("Ranked Features Based on Average Score Across Methods:")
print(df_sorted[['Feature', 'Average Score', 'Rank']])

# Plot feature importance for each method with the fix for the Seaborn deprecation warning
plt.figure(figsize=(10, 6))
sns.barplot(x='ANOVA', y='Feature', data=df_sorted, palette='Blues_r', hue=None, legend=False)
plt.title('Feature Importance Based on ANOVA F-Test')
plt.xlabel('ANOVA Score')
plt.ylabel('Feature')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Chi-Square', y='Feature', data=df_sorted, palette='Greens_r', hue=None, legend=False)
plt.title('Feature Importance Based on Chi-Square Test')
plt.xlabel('Chi-Square Score')
plt.ylabel('Feature')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Mutual Information', y='Feature', data=df_sorted, palette='coolwarm', hue=None, legend=False)
plt.title('Feature Importance Based on Mutual Information')
plt.xlabel('Mutual Information Score')
plt.ylabel('Feature')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='PCA (Explained Variance)', y='Feature', data=df_sorted, palette='Purples_r', hue=None, legend=False)
plt.title('Feature Importance Based on PCA')
plt.xlabel('PCA Explained Variance')
plt.ylabel('Feature')
plt.show()

# Plot the overall average scores for each feature
plt.figure(figsize=(10, 6))
sns.barplot(x='Average Score', y='Feature', data=df_sorted, palette='viridis', hue=None, legend=False)
plt.title('Average Feature Importance Across All Methods')
plt.xlabel('Average Score')
plt.ylabel('Feature')
plt.show()

# Incorporate External Feature Selection File (new feature selection file)
feature_selection_path = 'FEATURESELECTION_HD_new.xlsx'  # Adjust the path accordingly
feature_selection = pd.read_excel(feature_selection_path)

# Extract relevant columns from the external file
anova_feature_selection = feature_selection[['ANOVA TEST', 'Unnamed: 1']].dropna().rename(columns={'ANOVA TEST': 'Feature', 'Unnamed: 1': 'ANOVA'})
chi_square_feature_selection = feature_selection[['Chi-Square Test', 'Unnamed: 4']].dropna().rename(columns={'Chi-Square Test': 'Feature', 'Unnamed: 4': 'Chi-Square'})
mutual_info_feature_selection = feature_selection[['MUTUAL INFORMATION ', 'Unnamed: 7']].dropna().rename(columns={'MUTUAL INFORMATION ': 'Feature', 'Unnamed: 7': 'Mutual Information'})
pca_feature_selection = feature_selection[['PCA', 'Unnamed: 10']].dropna().rename(columns={'PCA': 'Feature', 'Unnamed: 10': 'PCA'})

# Merge all feature selection methods into one DataFrame from the external file
feature_selection_combined = pd.merge(anova_feature_selection, chi_square_feature_selection, on='Feature', how='outer')
feature_selection_combined = pd.merge(feature_selection_combined, mutual_info_feature_selection, on='Feature', how='outer')
feature_selection_combined = pd.merge(feature_selection_combined, pca_feature_selection, on='Feature', how='outer')

# Normalize the feature scores from the external file
score_columns_external = ['ANOVA', 'Chi-Square', 'Mutual Information', 'PCA']
feature_selection_combined[score_columns_external] = feature_selection_combined[score_columns_external].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Calculate the average score across all methods from the external file
feature_selection_combined['Average Score'] = feature_selection_combined[score_columns_external].mean(axis=1)

# Rank the features based on the average score
feature_selection_combined['Rank'] = feature_selection_combined['Average Score'].rank(ascending=False)

# Plot the average scores for each feature from the external feature selection file
plt.figure(figsize=(10, 6))
sns.barplot(x='Average Score', y='Feature', data=feature_selection_combined.sort_values(by='Rank'), palette='viridis', hue=None, legend=False)
plt.title('Average Feature Importance Across Methods from External Feature Selection File')
plt.xlabel('Average Score')
plt.ylabel('Feature')
plt.show()

# Ranked Features Based on Average Score Across Methods:
#         Feature  Average Score  Rank
# 7   cholesterol       0.594527   1.0
# 1           age       0.531912   2.0
# 5         ap_hi       0.352861   3.0
# 4        weight       0.267594   4.0
# 6         ap_lo       0.264856   5.0
# 0            id       0.251247   6.0
# 8          gluc       0.149878   7.0
# 2        gender       0.129766   8.0
# 3        height       0.100594   9.0
# 9         smoke       0.032337  10.0
# 11       active       0.021283  11.0
# 10         alco       0.015551  12.0
