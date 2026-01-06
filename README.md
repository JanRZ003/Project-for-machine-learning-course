# Project for machine learning course
Hierarchical taxonomy of malware families: Investigating behavioral relationships of PE malware via API action clustering

# Malware Behavioral Clustering Analysis

This project focuses on the hierarchical clustering of malware samples based on their behavioral profiles (API call sequences). The primary goal is to evaluate the effectiveness of various distance metrics and linkage methods in reconstructing known malware families.

## Source Data and Origin
The core methodology and initial ontology-based features are derived from the [PE-Malware-Ontology](https://github.com/orbis-security/pe-malware-ontology) repository.

Specifically, the following files originate from that research:
* **actions.jsonl**: Contains a predefined list of behavioral actions and API calls used as features.
* **ember_to_owl.py**: A script used to map raw EMBER data to higher-level behavioral concepts.

## Requirements and Setup

### Dataset Acquisition
Due to size limitations, the raw feature file is not included in this repository. To run the analysis, you must download the **EMBER 2018** dataset:
1. Visit the [EMBER GitHub repository](https://github.com/elastic/ember).
2. Follow the link in their README to the 2018 version.
3. Download and extract the file **train_features_1.jsonl**.
4. Ensure this file is placed in the same directory as the project scripts.

For the project to execute correctly, all scripts and data files must be located in the same root directory.

## Project Components
* **projekt.py**: The primary script developed for this research. It handles data loading, feature extraction, execution of hierarchical clustering (linkage), and calculation of validation metrics (ARI, Silhouette, Cophenetic).
* **vysledky_projektu/**: This directory contains the generated outputs of the script. For each configuration, a dendrogram is provided with its corresponding metrics (ARI, Silhouette Score, and Cophenetic Correlation Coefficient) included in the visualization.

## Authors
Jan RZ
