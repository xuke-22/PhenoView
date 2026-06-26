# PhenoView

A local, privacy-preserving platform for no-code exploration of phenotype datasets.

## Abstract

PhenoView is a lightweight, browser-based application for exploring high-dimensional phenotype datasets. Running entirely on the user’s own computer, it requires no programming expertise, no cloud upload, and no commercial software licenses, making phenotype data exploration accessible while preserving data privacy. From a single tidy CSV file, PhenoView provides integrated visualization, dimensionality reduction, correlation analysis, statistical testing, hierarchical clustering, and publication-ready figure export.

**Authors:** Ke Xu, Jesus Maria Gomez-Salinero  
Weill Cornell Medicine

## Installation

git clone https://github.com/xuke-22/PhenoView.git

cd PhenoView

pip install -r requirements.txt

## Running PhenoView

Navigate to the folder containing the application files and run:
```bash
streamlit run phenoview.py
```
This will open a local browser window with the PhenoView interface.

The application runs entirely locally.

No internet connection or cloud upload is required for routine use.

## Input Data Format

PhenoView expects a tidy CSV file where:
- each row represents one sample
- columns contain metadata or numeric features
- at least one column should define the group variable
- an optional Condition column can represent treatments
- an optional PairID column can define matched or paired samples for paired visualization and statistical analysis

Example structure:

| SampleID | Group | Condition | PairID | Feature1 | Feature2 |
|----------|------|-----------|--------|----------|----------|
| S1 | Control | Untreated | 1 | 1.2 | 5.4 |
| S2 | ERG/FLI1 | Untreated | 1 | 1.8 | 4.9 |
| S3 | Control | IL1/TNFa | 2 | 1.5 | 5.1 |
| S4 | ERG/FLI1 | IL1/TNFa | 2 | 2.0 | 4.7 |

If no Condition column is provided, analyses are performed using the Group variable only.

If no PairID column is present, paired visualization and paired statistical options are not shown.

## Main Features

PhenoView provides several integrated analysis views:
- Data preview and dataset validation
- PCA and UMAP dimensionality reduction
- Sample–sample and feature–feature correlation heatmaps
- Interactive plots (dot, bar, violin)
- Welch's t-test
- Optional Benjamini–Hochberg correction
- Optional paired analysis using PairID
- Z-scored heatmaps with hierarchical clustering
- Publication-ready PNG and SVG export
- Dynamic sample filtering
- Local, privacy-preserving analysis

## Dependencies

Main Python packages used:
•	streamlit
•	plotly
•	kaleido
•	pandas
•	numpy
•	scipy
•	scikit-learn
•	umap-learn
•	pillow

## Citation

If you use **PhenoView** in your research, please cite:

Xu K., Gomez-Salinero J.M.  
PhenoView: a local, privacy-preserving platform for no-code exploration of phenotype datasets.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
