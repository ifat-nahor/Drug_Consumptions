import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

## ============================================================================
## PART A: PHENOTYPE ANALYSIS (ANOVA)
## ============================================================================

def analyze_personality_profile(df, personality_cols):
    """
    Checks if 'Usage_Group' creates significantly different personality profiles.
    Uses ANOVA + Tukey Post-Hoc.
    """
    print("\n" + "="*80)
    print("🔬 PART A: NEURO-PROFILE ANALYSIS (ANOVA)")
    print("="*80)
    
    results = []
    
    # Run ANOVA for each trait
    for trait in personality_cols:
        groups = [df[df['Usage_Group'] == g][trait] for g in df['Usage_Group'].unique()]
        
        # F-Test
        f_stat, p_val = stats.f_oneway(*groups)
        
        significance = "NS"
        if p_val < 0.001: significance = "***"
        elif p_val < 0.01: significance = "**"
        elif p_val < 0.05: significance = "*"
            
        results.append({
            'Trait': trait,
            'F-Statistic': f_stat,
            'P-Value': p_val,
            'Sig': significance
        })

    results_df = pd.DataFrame(results).set_index('Trait')
    print(results_df)
    return results_df


## ============================================================================
## PART B: PREDICTIVE MODELING TOURNAMENT (ML)
## ============================================================================

def compare_prediction_models(df, personality_cols):
    """
    Runs a competition between Logistic Regression (Linear) and Random Forest (Non-Linear)
    to see which model better predicts 'Regular User' status.
    """
    print("\n" + "="*80)
    print("🤖 PART B: MODEL TOURNAMENT (LOGISTIC REGRESSION vs RF)")
    print("="*80)
    
    # Prepare Data: Binary Classification (Regular vs Non-User)
    binary_df = df[df['Usage_Group'].isin(['Non-User', 'Regular'])].copy()
    binary_df['Target'] = binary_df['Usage_Group'].apply(lambda x: 1 if x == 'Regular' else 0)
    
    X = binary_df[personality_cols]
    y = binary_df['Target']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 1. Logistic Regression
    log_model = LogisticRegression(class_weight='balanced', max_iter=1000)
    log_model.fit(X_train, y_train)
    acc_log = accuracy_score(y_test, log_model.predict(X_test))
    
    # 2. Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)
    acc_rf = accuracy_score(y_test, rf_model.predict(X_test))
    
    print(f"📊 Accuracy Results:")
    print(f"   Logistic Regression: {acc_log:.2%}")
    print(f"   Random Forest:       {acc_rf:.2%}")
    
    # Feature Importance Plot
    importances = pd.DataFrame({
        'Feature': personality_cols,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importances, palette='viridis')
    plt.title('Feature Importance (Random Forest)', fontsize=15)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()
    
    return importances


## ============================================================================
## PART C: HYPOTHESIS TESTING (FISHER Z)
## ============================================================================

def compare_correlations_fisher_z(df):
    """
    Tests if the correlation between Impulsivity and SS is statistically 
    different between Non-Users and Regular Users.
    """
    print("\n" + "="*80)
    print("📉 PART C: STATISTICAL MODERATION CHECK (FISHER Z)")
    print("="*80)
    
    groups = ['Non-User', 'Regular']
    r_values, n_values = [], []
    
    for g in groups:
        sub = df[df['Usage_Group'] == g]
        if len(sub) < 3: continue # Safety check
        r, _ = stats.pearsonr(sub['Impulsive'], sub['SS'])
        r_values.append(r)
        n_values.append(len(sub))
        print(f"Group {g}: r = {r:.3f} (n={len(sub)})")
        
    # Fisher Z Calculation
    z1 = 0.5 * np.log((1+r_values[0])/(1-r_values[0]))
    z2 = 0.5 * np.log((1+r_values[1])/(1-r_values[1]))
    se_diff = np.sqrt((1/(n_values[0]-3)) + (1/(n_values[1]-3)))
    z_score = (z1 - z2) / se_diff
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    print(f"\nDifference Z-Score: {z_score:.3f}")
    print(f"P-Value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("✅ SIGNIFICANT difference in correlation strength.")
    else:
        print("❌ NO significant difference found.")


## ============================================================================
## PART D: CROSS-FAMILY ANALYSIS (NEW!)
## ============================================================================

def compare_drug_families_analysis(df):
    """
    Compares the Impulsivity-SS correlation across the 3 drug families.
    Uses the 'Score_...' columns created in preprocessing.
    """
    print("\n" + "="*80)
    print("💊 PART D: DRUG FAMILY COMPARISON")
    print("="*80)
    
    families = {
        'Stimulants': 'Score_Stimulants',
        'Depressants': 'Score_Depressants',
        'Hallucinogens': 'Score_Hallucinogens'
    }
    
    results = []
    
    for name, col in families.items():
        if col not in df.columns: continue
        
        # We analyze "Heavy Users" of this specific family (Score >= 5)
        heavy_users = df[df[col] >= 5]
        
        if len(heavy_users) > 10:
            r, p = stats.pearsonr(heavy_users['Impulsive'], heavy_users['SS'])
            results.append({'Family': name, 'Correlation': r, 'P-Value': p})
            print(f"{name} (Heavy Users): r={r:.3f}, p={p:.4f}")
            
    # Visualization
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        plt.figure(figsize=(8, 5))
        sns.barplot(x='Family', y='Correlation', data=res_df, palette='magma')
        plt.title('Correlation Strength by Drug Family', fontsize=14)
        plt.ylim(0, 1)
        plt.savefig('family_comparison.png')
        plt.show()
        
    return res_df