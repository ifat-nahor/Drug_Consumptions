import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os

def run_flexible_ttest_and_trend(df, target_drug_col, trait_cols, threshold=3):
    """
    מבחן T וגרף מגמה

    פרמטרים:
    df: ה-DataFrame שלך
    target_drug_col: שם העמודה של הסם (ערכים 0-6)
    trait_cols: רשימת עמודות של תכונות אישיות (כמו BIS11, ImpSS)
    threshold: הערך שממנו ומעלה אדם נחשב "משתמש" (ברירת מחדל 3 = שימוש בשנה האחרונה)
    """
    
    print(f"\n" + "="*50)
    print(f"ANALYSIS FOR: {target_drug_col}")
    print("="*50)

    # 1. יצירת קבוצות למבחן T (User vs Non-User)
    # אנחנו יוצרים עותק כדי לא לפגוע בדאטה המקורי
    temp_df = df.copy()
    temp_df['Group'] = temp_df[target_drug_col].apply(
        lambda x: 'User' if x >= threshold else 'Non-User'
    )

    # ביצוע מבחן T לכל תכונה
    for trait in trait_cols:
        group_u = temp_df[temp_df['Group'] == 'User'][trait].dropna()
        group_nu = temp_df[temp_df['Group'] == 'Non-User'][trait].dropna()
        
        if len(group_u) > 1 and len(group_nu) > 1:
            t_stat, p_val = stats.ttest_ind(group_u, group_nu)
            status = "Significant" if p_val < 0.05 else "Not Significant"
            print(f"Trait: {trait:10} | p-value: {p_val:.4e} ({status})")
        else:
            print(f"Trait: {trait:10} | Not enough data for T-test")

    # 2. יצירת גרף המגמה (The Trend Graph)
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # הכנת הדאטה לגרף (Melt)
    df_plot = temp_df.melt(id_vars=[target_drug_col], value_vars=trait_cols, 
                           var_name='Trait', value_name='Score')

    # ציור קווי המגמה
    plot = sns.pointplot(x=target_drug_col, y='Score', hue='Trait', 
                         data=df_plot, markers=["o", "s", "D"], linestyles=["-", "--", "-."],
                         capsize=.1, palette="husl")

    # הגדרת תוויות לציר ה-X (מותאם ל-CL0-CL6)
    labels = ['Never', 'Decade', 'Year', 'Month', 'Week', 'Day', 'Daily']
    plt.xticks(ticks=range(len(labels)), labels=labels[:len(temp_df[target_drug_col].unique())])

    plt.title(f'Trend Analysis: Personality vs. {target_drug_col} Usage', fontsize=16)
    plt.xlabel('Frequency of Consumption', fontsize=12)
    plt.ylabel('Average Trait Score', fontsize=12)
    plt.legend(title='Personality Traits')
    
    # שמירה
    plt.tight_layout()
    file_name = f'trend_{target_drug_col.lower()}.png'
    plt.savefig(file_name)
    print(f"✓ Graph saved as: {file_name}")
    plt.show()

# --- דוגמה להפעלה על הדאטה שלך ---
# נניח ש-df_cleaned הוא הדאטה שלך:
my_traits = ['BIS11', 'ImpSS'] # אפשר להוסיף כאן עוד עמודות נומריות
run_flexible_ttest_and_trend(df_cleaned, target_drug_col='Coke', trait_cols=my_traits)

# אפשר להריץ את זה בלולאה על כל הסמים בלחיצה אחת:
# for drug in drug_columns:
#     run_flexible_ttest_and_trend(df_cleaned, drug, my_traits)