# Movie-Reccomendation-System
## Data Preprocessing

This Project focuses on preparing the dataset so it is clean, consistent, and ready for modeling. The goal is to produce a reproducible preprocessing pipeline and document each step clearly.

---

## 1) Data Preprocessing

### 1.1 Data Cleaning
**Objective:** Improve data quality by handling missing, inconsistent, duplicate, or noisy records.

**Planned/Implemented Tasks**
- **Missing values**
  - Detect missing/blank/null values per column.
  - Strategy (choose based on column type):
    - Numerical: impute with mean/median or use model-based imputation.
    - Categorical: impute with mode or “Unknown”.
    - Critical fields: remove rows if missing values make the record unusable.
- **Duplicates**
  - Detect duplicate rows / duplicate IDs (e.g., movieId, userId + movieId pairs).
  - Remove duplicates while preserving the latest/most reliable entry.
- **Outliers & invalid values**
  - Detect impossible values (e.g., ratings outside allowed range).
  - Address outliers via capping, removal, or robust scaling depending on impact.
- **Text cleanup (if applicable)**
  - Normalize text fields (trim spaces, lowercase, consistent separators).
  - Standardize genres/tags (e.g., split by `|`, remove stray characters).

**Deliverables**
- Summary table: missing values count by column.
- Number of duplicates removed.
- Any rules applied to invalid/outlier values.

---

### 1.2 Data Integration
**Objective:** Combine additional data sources to enrich the dataset.

**Examples **
- Merge movie metadata (genres, year, cast/crew, plot keywords).
- Include external ratings or popularity signals.
- Integrate user demographic information (if available).

**Deliverables (if performed)**
- List of sources integrated.
- Join keys used (e.g., `movieId`, `imdbId`, `title + year`).
- Row counts before/after merges and any dropped unmatched rows.

---

### 1.3 Data Reduction (Implement techniques where necessary)
**Objective:** Reduce dataset size or dimensionality while preserving the most useful information.

You may implement **one or more** of the following techniques (as appropriate for the dataset):

#### A) PCA (Principal Component Analysis)
- Use PCA after scaling features (important for distance-based / variance-based methods).
- Choose number of components based on explained variance (e.g., 90–95%).
- **Deliverables**
  - Explained variance plot or explained variance table.
  - Components selected and rationale.

#### B) Attribute Subset Selection
- Remove low-utility or redundant features using:
  - Correlation analysis
  - Mutual information
  - Feature importance (tree-based models)
  - Domain knowledge (e.g., removing identifiers that leak)
- **Deliverables**
  - Features removed and justification.

#### C) Regression-Based Reduction (Optional / If relevant)
These are useful when predicting a continuous target (e.g., rating prediction).
- **Linear Regression**
- **Multiple Regression**
- **Log-Linear Model**
- **Deliverables**
  - Selected predictors and performance metrics (RMSE/MAE/R²) if used for feature selection.

#### D) Histogram Analysis
- Inspect distributions to detect skew, sparsity, and outliers.
- **Deliverables**
  - Key histograms (ratings distribution, user activity, movie popularity).

#### E) Clustering (for reduction or grouping)
- Cluster users/movies to reduce complexity or create segments.
- Algorithms: K-Means / Hierarchical / DBSCAN (depending on data).
- **Deliverables**
  - Cluster count selection method (elbow/silhouette).
  - Cluster interpretation.

#### F) Sampling
- Downsample large datasets for faster prototyping (while keeping representativeness).
- Stratified sampling if there is class imbalance (for classification tasks).
- **Deliverables**
  - Sampling ratio + method used.

#### G) Data Compression
- Reduce storage/memory usage:
  - Use sparse matrices for user-item interactions.
  - Reduce numeric dtype sizes where safe.
- **Deliverables**
  - Memory usage before/after (if measured).

---

### 1.4 Data Transformation
**Objective:** Transform data into a modeling-ready format.

#### A) Normalization / Scaling
- Common approaches:
  - Min-Max scaling (0–1)
  - Standardization (z-score)
  - Robust scaling (median/IQR) for outlier-heavy data
- **Deliverables**
  - Which columns were scaled and method used.

#### B) Data Discretization
- Convert continuous variables into bins if needed (e.g., user activity level: low/medium/high).
- Approaches:
  - Equal-width binning
  - Equal-frequency binning
- **Deliverables**
  - Binning strategy and bin edges.

---

## 2) Modeling Requirement (Next Phase)
After preprocessing, we will apply **one** of the following:

### 1: Classification
Example framing:
- Predict whether a user will like a movie (like/dislike) based on features.
- Models could include Logistic Regression, Random Forest, SVM, etc.

### 2: Cluster Analysis
Example framing:
- Group users by viewing/rating behavior.
- Group movies by content/genre/tag similarity.

## 3) Report Requirement
A report will be prepared documenting:
- Dataset description and sources
- Preprocessing steps (cleaning, integration, reduction, transformation)
- Key findings (plots/statistics)
- Team contributions (who did what)
- Next steps for modeling

---

## 4) Team Contribution
Each team member must contribute to:
- Code (commits / notebooks / scripts)
- Documentation (README + report sections)
- Validation (reviewing results, checking assumptions)

We will track contributions via Git commit history and a contribution table in the report.

---

## How to Run (To be updated)
Add your exact commands once scripts/notebooks are finalized:
- Install dependencies
- Run preprocessing
- Export processed dataset / features
