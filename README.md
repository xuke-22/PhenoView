# PhenoView

Interactive application for exploration and visualization of phenotype datasets.

**Authors:** Ke Xu, Jesus Maria Gomez-Salinero  
Weill Cornell Medicine

PhenoView is a lightweight Python application that allows users to explore high-dimensional phenotype datasets through an interactive browser interface. It provides dimensionality reduction, correlation analysis, group comparisons, and heatmap visualization without requiring programming experience.

The application runs locally using Streamlit and generates publication-ready figures using Plotly.

# Installation

Clone or download this repository.

Install the required Python packages:
```bash
pip install -r requirements.txt
```

# Running PhenoView

Navigate to the folder containing the application files and run:
```bash
streamlit run phenoview.py
```
This will open a local browser window with the PhenoView interface.

The application runs locally and does not require an internet connection.

# Input Data Format

PhenoView expects a tidy CSV file where:
- each row represents one sample
- columns contain metadata or numeric features
- at least one column should define the group variable
- an optional Condition column can represent treatments

Example structure:

| SampleID | Group | Condition | Feature1 | Feature2 |
|----------|------|-----------|----------|----------|
| S1 | Control | Untreated | 1.2 | 5.4 |
| S2 | Control | IL1/TNFa | 1.5 | 5.1 |
| S3 | ERG/FLI1 | Untreated | 2.0 | 4.7 |

If no Condition column is provided, analyses are performed using the Group variable only.

# Main Features

PhenoView provides several integrated analysis views:
- Data preview and dataset validation
- PCA and UMAP dimensionality reduction
- Sample–sample and feature–feature correlation heatmaps
- Interactive plots (dot, bar, violin) with Welch’s t-test
- Z-scored heatmaps with hierarchical clustering
- Optional paired visualization when a pairing column is detected
- Export of figures (PNG/SVG)

# Dependencies

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

# Citation

If you use **PhenoView** in your research, please cite:

Xu K., Gomez-Salinero J.M.  
PhenoView: an interactive application for exploration and visualization of phenotype datasets.

# License

PhenoView is provided for academic research use.
