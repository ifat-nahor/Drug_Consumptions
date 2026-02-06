link to paper - https://docs.google.com/document/d/1K6MS1hrZCL0JNNjQYB6MfY3OizG4ImdhyQM_u0x6mvU/edit?usp=sharing

Project Overview

first Research Question:
To what extent do specific personality traits—specifically Impulsivity and Sensation Seeking (SS)—predict the frequency of drug consumption across distinct neuro-pharmacological categories (Hard Drugs, Psychedelics, Prescription, and Daily Addictive substances)?

Hypothesis:We hypothesize a significant positive correlation between high levels of Impulsivity/SS and increased substance use frequency. 

 Conclusions: 
 Personality traits are significant predictors for 17 out of 19 drugs analyzed.
 Higher substance use frequency is directly linked to higher Impulsivity and Sensation Seeking scores.
 The strongest personality-usage links were found in Hard Drugs and Psychedelics categories.
 Results remained significant even after applying a strict Bonferroni correction.

second Research Question: Can drug usage be predicted based on personality profiles?

Hypothesis: Personality profiles (derived from Big Five traits) will significantly predict frequency of use across 19 different drug types (ANOVA, p<0.05).

 Conclusions: examining the relationship between personality traits (Nscore, Escore, Oscore, Ascore, Cscore) and drug consumption patterns using PCA dimensionality reduction and KMeans clustering - we found that we can predict drug usage based on personality profiles.


Instructions for Running the Project

# 1️. Set up Python virtual environment
python -m venv .venv

# Activate the environment
# Windows:
.venv\Scripts\activate
# Linux / Mac:
source .venv/bin/activate

# 2️.Install required packages (make sure you are in the project folder)
pip install -r requirements.txt

# 3️.Run the main project workflow
python main.py

# 4️.Check outputs
# - All plots are saved in plot_pics_for_groups_vs_imp_ss
# - PCA visualizations saved in plots_pics_for_pca_cluster/
# - Execution logs are printed directly to the console

# 5️.Run tests (optional)
pytest tests/






