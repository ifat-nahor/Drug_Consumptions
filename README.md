
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

Neuroscience data analysis project examining the relationship between personality traits (Nscore, Escore, Oscore, Ascore, Cscore) and drug consumption patterns using PCA dimensionality reduction and KMeans clustering on 1885 participants.

 Instructions for Running the Project

# 1️. Set up Python virtual environment
python -m venv .venv

# Activate the environment
# Windows:
.venv\Scripts\activate
# Linux / Mac:
source .venv/bin/activate

# 2️.Install required packages
pip install -r requirements.txt

# 3️.Run the main project workflow
python main.py

# 4️.Check outputs
# - All plots are saved in plot_pics/
# - PCA visualizations saved in plots_pics_for_pca_cluster/
# - Logs are in data_cleaning.log

# 5️.Run tests (optional)
pytest tests/










## Project Description
[cite_start]This project analyzes the relationship between personality traits—specifically **Impulsivity (BIS-11)** and **Sensation Seeking (ImpSS)**—and the frequency of drug and alcohol consumption[cite: 127]. 

### Objectives & Hypothesis
* [cite_start]**Main Objective**: To determine if the correlation between Impulsivity and Sensation Seeking varies across different levels of drug usage[cite: 127].
* [cite_start]**Hypothesis**: We expect to find a significant positive correlation (r > 0.5, p < 0.05) between these traits, with the strength of the relationship being more pronounced in regular users[cite: 137, 138].

## Data Description
[cite_start]The analysis is based on the **Drug Consumption Classification System** dataset[cite: 73].
* [cite_start]**Participants**: 1,885 individuals[cite: 73].
* [cite_start]**Variables**: Includes 12 personality traits (NEO-FFI-R, BIS-11, ImpSS), demographic data, and consumption records for 18 substances[cite: 74, 75].
* **Dataset Link**: [Kaggle - Drug Consumption UCI](https://www.kaggle.com/datasets/obeykhadija/drug-consumptions-uci)

## Project Structure
* [cite_start]`main.py`: The entry point of the project, orchestrating the analysis workflow[cite: 40].
* [cite_start]`requirements.txt`: List of necessary Python libraries[cite: 22].
* `drug_consumption.csv`: The raw data file.
* [cite_start]`README.md`: Project documentation and instructions[cite: 25].

## Key Stages of Analysis
1. [cite_start]**Data Loading & Cleaning**: Importing the dataset and assigning meaningful headers[cite: 30].
2. [cite_start]**Preprocessing**: Converting categorical usage labels (CL0-CL6) into numerical values for statistical calculation[cite: 120].
3. [cite_start]**Group Classification**: Categorizing users into "Regular", "Occasional", and "Non-Users"[cite: 129].
4. [cite_start]**Statistical Analysis**: Calculating Pearson correlation coefficients within each group[cite: 133].
5. [cite_start]**Clustering**: Applying K-Means to identify personality-based profiles among participants[cite: 125].
6. [cite_start]**Visualization**: Generating regression plots to illustrate the findings[cite: 135].

## Installation & Running Instructions
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt


   