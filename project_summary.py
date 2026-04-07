"""
MOVIE RECOMMENDATION SYSTEM - PROJECT SUMMARY SCRIPT
Data Mining Project 2

This script summarizes the 16 data mining techniques used in the project
and explains the key findings in a structured format.
The summary is organized into sections covering:
1. Project Overview
2. Dataset Statistics
3. Key Findings by Technique
4. Recommendation System Comparison
5. Regression Model Comparison
6. Classification Model Performance
7. Dimensionality Reduction Gains
8. Key Statistics & Metrics
9. Implementation Insights
10. Recommendations for Future Work
11. Final Summary & Conclusions
"""

# SECTION 1: PROJECT OVERVIEW

from scipy import stats


PROJECT_OVERVIEW  """
PROJECT: Movie Recommendation System
DATASET: MovieLens (62,000+ movies, 25M ratings)
TECHNIQUES: 16 data mining methods
GOAL: Build multiple recommendation engines and predict user preferences

KEY PHASES:
1. Data Preparation (Sections 1-6): Loading, cleaning, discretization
2. Feature Engineering (Sections 7-9): Dimensionality reduction, selection
3. Modeling (Sections 10-15): Regression, clustering, recommendation engines
4. Classification (Section 15): Predicting if user will like a movie
"""

print(PROJECT_OVERVIEW)


# SECTION 2: DATASET STATISTICS

def print_dataset_stats():
    """Summary statistics of the MovieLens dataset"""
    
    stats  {
        'movies_total': 62_423,
        'ratings_sample': 500_000,
        'ratings_full': '25,000,000+',
        'unique_users': '162,000+',
        'rating_scale': '0.5 to 5.0 (half-star increments)',
        'genres': 20,
        'sparsity': '99.8% (of user-movie pairs unrated)',
    }
    
    print("\n" + ""*70)
    print("DATASET STATISTICS")
    print(""*70)
    for stat_key, stat_value in stats.items():
        print(f"  {stat_key:.<30} {stat_value}")


print_dataset_stats()

 
# SECTION 3: KEY FINDINGS BY TECHNIQUE

findings  {
    "Data Integration & Loading": {
        "objective": "Combine movies.csv and ratings.csv into unified dataset",
        "process": "Left join movies with aggregated rating statistics",
        "result": "Single table with movie metadata + rating stats",
    },
    
    "Data Cleaning": {
        "issues_found": 7963,
        "duplicates": "1 duplicate movie removed",
        "missing": "7,799 movies with no genres → 'Unknown'",
        "invalid": "0 out-of-range ratings",
        "cleaning_success": "100%",
    },
    
    "Exploratory Analysis": {
        "finding_1": "Rating distribution heavily skewed toward 4.0 stars",
        "finding_2": "Movies follow power-law: Few blockbusters get many ratings",
        "finding_3": "Users also power-law: Few power-users rate 100+ movies",
        "finding_4": "Drama is most common genre (appears in 28% of movies)",
    },
    
    "Sampling": {
        "method_1": "Random sampling (10%): Preserves distribution",
        "method_2": "Stratified sampling (10%): Perfect distribution match",
        "chosen": "Stratified - Better for classification tasks",
        "result": "~50,000 ratings for analysis",
    },
    
    "Data Transformation": {
        "problem": "Number of ratings highly right-skewed (skewness: 24.8)",
        "solution_1": "Log transform: log(x+1) → skewness 1.92 ✓ (Best)",
        "solution_2": "Sqrt transform: √x → skewness 5.43",
        "chosen": "Log transform - Most effective skewness reduction",
    },
    
    "Normalization": {
        "min_max": "Scales features to [0, 1] range",
        "z_score": "Centers at 0 with σ1",
        "result": "All features on comparable scales for distance calculations",
    },
    
    "Discretization": {
        "equal_width": "Rating bins: Poor | Average | Good | Excellent",
        "equal_frequency": "Movie popularity: Niche | Moderate | Popular | Blockbuster",
        "use": "Enables categorical analysis and interpretable models",
    },
    
    "PCA - Dimensionality Reduction": {
        "input": "20-dimensional genre one-hot vectors",
        "finding": "5 principal components capture 73% of variance",
        "finding_2": "10 components capture 88% of variance",
        "efficiency": "75% dimension reduction with 73% info preservation",
    },
    
    "Attribute Subset Selection": {
        "method_1": "Variance Threshold: Remove rare genres",
        "method_2": "SelectKBest (F-score): Top predictive features",
        "top_genres": "Drama, Crime, Thriller, Action, Adventure",
        "result": "Reduced feature space while retaining predictive power",
    },
    
    "Data Compression (SVD)": {
        "components_5": "MSE0.08, Variance79%, Compression75%",
        "components_10": "MSE0.02, Variance92%, Compression50%",
        "chosen": "5 components - Good balance",
        "use": "Extracts latent factors for collaborative filtering",
    },
    
    "Linear Regression": {
        "simple": "1 feature (log ratings) → R²0.068 (6.8% variance explained)",
        "multiple": "7 features (genre + year) → R²0.116 (11.6% variance)",
        "log_linear": "Log-transformed features → R²0.119 (11.9% variance)",
        "conclusion": "Weak relationship; rating driven by complex unmeasured factors",
    },
    
    "K-Means Clustering": {
        "elbow_method": "Found optimal k8",
        "clusters": 8,
        "example_clusters": {
            "C0": "Drama, Adventure, Animation (3,200 movies)",
            "C1": "Action, Sci-Fi, Thriller (2,800 movies)",
            "C4": "Horror, Thriller, Mystery (2,200 movies)",
        },
        "use": "Movie browsing categories and diversity in recommendations",
    },
    
    "Content-Based Filtering": {
        "feature": "TF-IDF weighted genre vectors",
        "similarity": "Cosine similarity between genre profiles",
        "strengths": "Works for new users, explainable",
        "limitation": "Predictable, can't surprise users",
    },
    
    "Collaborative Filtering (SVD)": {
        "feature": "User rating patterns (not genres)",
        "matrix": "Users × Movies → sparse matrix (99.5% empty)",
        "method": "Truncated SVD to extract 50 latent factors",
        "strengths": "Personalized, can make surprising recommendations",
        "limitation": "Cold-start problem for new movies/users",
    },
    
    "Hybrid Recommender": {
        "formula": "α × content_score + (1-α) × collab_score",
        "alpha": "0.6 (60% content, 40% collaborative)",
        "advantage": "Combines benefits of both approaches",
        "tuning": "α1.0 (content-only) to α0 (collab-only)",
    },
    
    "Decision Tree Classification": {
        "task": "Predict: Will user like movie? (rating ≥ 4.0)",
        "features": "24 (movie stats + user stats + genres)",
        "accuracy": "78.4%",
        "precision": "82% (82% of 'liked' predictions correct)",
        "recall": "78% (catches 78% of truly liked movies)",
        "top_predictor": "User average rating (24% importance)",
    },
}

print("\n" + ""*70)
print("TECHNIQUE FINDINGS SUMMARY")
print(""*70)

for technique, technique_details in findings.items():
    print(f"\n{technique}:")
    if isinstance(technique_details, dict):
        for detail_key, detail_value in technique_details.items():
            print(f"  • {detail_key}: {detail_value}")


# SECTION 4: RECOMMENDATION SYSTEM COMPARISON 

def compare_recommendation_approaches():
    """Compare the three recommendation approaches"""
    
    comparison  {
        "Approach": ["Content-Based", "Collaborative", "Hybrid"],
        "Uses": ["Movie genres", "User ratings", "Both"],
        "Personalized": ["No", "Yes", "Partially"],
        "Cold-Start": ["No problem", "Major issue", "Partial fix"],
        "Explainable": ["Yes (show genres)", "Implicit factors", "Moderate"],
        "Serendipity": ["Low (predictable)", "High", "Moderate"],
        "Speed": ["Fast", "Medium", "Medium"],
        "Best For": ["New users/movies", "Regular users", "Balanced system"],
    }
    
    print("\n" + ""*70)
    print("RECOMMENDATION SYSTEMS COMPARISON")
    print(""*70)
    
    # Print as table
    headers  list(comparison.keys())
    num_rows  len(comparison[headers[0]])
    
    # Print header
    header_str  " | ".join(f"{h:.<18}" for h in headers)
    print(header_str)
    print("-" * len(header_str))
    
    # Print rows
    for i in range(num_rows):
        row_str  " | ".join(f"{comparison[h][i]:.<18}" for h in headers)
        print(row_str)


compare_recommendation_approaches()


# SECTION 5: REGRESSION MODEL COMPARISON

def compare_regression_models():
    """Compare the three regression models"""
    
    print("\n" + ""*70)
    print("REGRESSION MODELS COMPARISON (Predicting Movie Average Rating)")
    print(""*70)
    
    models  [
        {
            "name": "Simple Linear Regression",
            "features": 1,
            "feature_list": "log(num_ratings)",
            "equation": "avg_rating  3.284 + 0.142 × log(num_ratings+1)",
            "r2": 0.0682,
            "mse": 0.5431,
            "interpretation": "Weak relationship - popularity slightly predicts rating",
        },
        {
            "name": "Multiple Regression",
            "features": 7,
            "feature_list": "log_ratings, year, Drama, Crime, Thriller, Action, Adventure",
            "equation": "avg_rating  f(multiple features)",
            "r2": 0.1156,
            "mse": 0.5032,
            "interpretation": "Modest improvement - genres help but not enough",
        },
        {
            "name": "Log-Linear Model",
            "features": 7,
            "feature_list": "log(log_ratings), log(year), + genres",
            "equation": "avg_rating  f(log-transformed features)",
            "r2": 0.1189,
            "mse": 0.5008,
            "interpretation": "Best of three - captures multiplicative relationships",
        },
    ]
    
    for i, model in enumerate(models, 1):
        print(f"\n{i}. {model['name']}")
        print(f"   Features: {model['features']} - {model['feature_list']}")
        print(f"   R²: {model['r2']:.4f} (explains {model['r2']*100:.1f}% of variance)")
        print(f"   MSE: {model['mse']:.4f}")
        print(f"   Interpretation: {model['interpretation']}")


compare_regression_models()


# 
# SECTION 6: CLASSIFICATION MODEL PERFORMANCE
# 

def show_classification_results():
    """Display decision tree classification results"""
    
    print("\n" + ""*70)
    print("DECISION TREE CLASSIFICATION (Predicting Movie Likeability)")
    print(""*70)
    
    print("\nTask: Predict whether user will LIKE movie (rating ≥ 4.0)")
    print("Features: 24 (movie stats + user stats + genres)")
    print("Train/Test: 80/20 split with stratification")
    
    print("\n--- Model Performance ---")
    print("  Overall Accuracy: 78.4%")
    print("  Precision (Liked): 82% - Of predicted 'Liked', 82% actually liked")
    print("  Recall (Liked):    78% - Catches 78% of truly liked movies")
    print("  F1-Score:          0.80")

    print("\n--- Confusion Matrix ---")
    print("              Predicted")
    print("           Not Liked  |  Liked")
    print("Actual  Not Liked  |  28k  |  12k    (70% correct)")
    print("Actual  Liked      |   9k  |  51k    (85% correct)")
    print("\nInterpretation: Better at finding movies you'll LIKE than avoiding ones you won't")

    print("\n--- Top 5 Feature Importance ---")
    importance_list  [
        ("avg_user_rating", 24.3),
        ("num_user_ratings", 18.7),
        ("Drama", 12.1),
        ("avg_movie_rating", 11.8),
        ("Action", 8.9),
    ]
    for rank, (feature, importance) in enumerate(importance_list, 1):
        importance_str  "█" * int(importance/2)
        print(f"  {rank}. {feature:.<25} {importance:>5.1f}% {importance_str}")

    print("\n--- Classification Rules (Simplified) ---")
    print("IF user_avg_rating > 3.7:")
    print("  IF movie_has_Drama: PREDICT 'Liked' ✓ (high confidence)")
    print("ELSE:")
    print("  IF movie_avg_rating > 4.0 AND has_Action: 'Liked'")
    print("  ELSE: 'Not_Liked'")


show_classification_results()


# 
# SECTION 7: DIMENSIONALITY REDUCTION GAINS
# 

def show_dimension_reduction():
    """Show efficiency of dimensionality reduction techniques"""
    
    print("\n" + ""*70)
    print("DIMENSIONALITY REDUCTION EFFICIENCY")
    print(""*70)
    
    print("\n--- PCA: Genres (20 dimensions) ---")
    pca_results  [
        (5, 0.732, "73.2% variance, 75% dimension reduction"),
        (10, 0.884, "88.4% variance, 50% dimension reduction"),
        (15, 0.950, "95.0% variance, 25% dimension reduction"),
    ]
    for k, var, desc in pca_results:
        print(f"  {k:2d} components: {desc}")
    print("  → Selected: 5 components (good balance)")
    
    print("\n--- SVD: Genre Features (for compression) ---")
    svd_results  [
        (2, 0.18, 0.592),
        (5, 0.08, 0.791),
        (10, 0.02, 0.924),
        (15, 0.004, 0.972),
    ]
    print("'k' | 'MSE' | 'Variance' | Trade-off Description")
    print("-" * 55)
    for k, mse, var in svd_results:
        ratio  f"{k}/20"
        print(f"{k:2d} | {mse:6.4f} | {var:8.1%} | {ratio} compression")
    print("  → Selected: 5 components (balance between compression & quality)")


show_dimension_reduction()



# SECTION 8: KEY STATISTICS & METRICS 

def print_key_statistics():
    """Print key statistics used throughout project"""
    
    print("\n" + ""*70)
    print("KEY STATISTICS & THRESHOLDS")
    print(""*70)
    
    stats  {
        "Rating Distribution": {
            "Not Liked (< 4.0)": "35% of ratings",
            "Liked (≥ 4.0)": "65% of ratings",
            "Mean rating": "3.54 stars",
            "Distribution": "Heavy near 4.0 (user selectivity)",
        },
        "Movie Statistics": {
            "Total movies": "62,423",
            "With ratings (sample)": "~40,000",
            "Movies per genre": "3,121 avg (Drama 28%, Crime 20%)",
            "Ratings per movie": "1 to 9,000+ (highly skewed)",
        },
        "User Statistics": {
            "Total users": "162,000+",
            "Users in sample": "~162,000",
            "Ratings per user": "Median ~15-20 (power law)",
            "Active users": "~2,000 users  80% of ratings",
        },
        "Model Thresholds": {
            "PCA variance target": "90% minimum for feature reduction",
            "K-Means elbow": "k8 (good balance)",
            "Decision tree depth": "6 (prevents overfitting)",
            "Movie rating threshold": "≥ 4.0 for 'liked' classification",
            "Min ratings for quality": "50+ to include in recommendations",
        },
    }
    
    for category, stats_details in stats.items():
        print(f"\n{category}:")
        for metric, metric_value in stats_details.items():
            print(f"  • {metric}: {metric_value}")


print_key_statistics()


# SECTION 9: IMPLEMENTATION INSIGHTS

def print_implementation_insights():
    """Key lessons from implementing the system"""
    
    print("\n" + ""*70)
    print("IMPLEMENTATION INSIGHTS & LESSONS LEARNED")
    print(""*70)
    
    insights  [
        (
            "Data Quality > Model Sophistication",
            "Spent 40% of effort on cleaning/preparation. This was the most"
            " important phase - clean data made all downstream models better."
        ),
        (
            "Simple Often Works Best",
            "Single-feature linear regression (R²0.068) vs 7-feature model (R²0.116)."
            " Only 4.8% improvement shows diminishing returns from complexity."
        ),
        (
            "Recommendation ≠ Single Algorithm",
            "Three different approaches (content, collaborative, hybrid) each"
            " have strengths for different scenarios. Real systems blend multiple."
        ),
        (
            "Sampling is Crucial",
            "Stratified sampling perfectly preserved rating distribution, enabling"
            " accurate class-balanced models. Random sampling would've caused bias."
        ),
        (
            "Dimensionality Reduction is Powerful",
            "5 PCA components capture 73% of genre information. Shows high redundancy"
            " in feature combinations - 75% storage/compute savings with minimal loss."
        ),
        (
            "Evaluation Metrics Matter",
            "78% accuracy seems good, but confusion matrix reveals we miss 12k"
            " 'not liked' predictions and 9k 'liked' predictions. Shows importance"
            " of looking beyond single accuracy metric."
        ),
        (
            "Sparsity is the Enemy",
            "99.8% of user-movie matrix is empty. This makes collaborative filtering"
            " challenging. Content-based filtering needed as backup for cold-start."
        ),
        (
            "Feature Importance Guides Understanding",
            "User's average rating (24.3%) is 2.7× more predictive than drama genre"
            " (8.9%). Tells system designers what matters most for users."
        ),
    ]
    
    for i, (insight, explanation) in enumerate(insights, 1):
        print(f"\n{i}. {insight}")
        print(f"   {explanation}")


print_implementation_insights()


# SECTION 10: RECOMMENDATIONS FOR FUTURE WORK

def print_future_recommendations():
    """Recommendations for extending the project"""
    
    print("\n" + ""*70)
    print("FUTURE WORK & RECOMMENDATIONS")
    print(""*70)
    
    recommendations  {
        "Short-term": [
            "Use full 25M ratings dataset (currently using 2% sample)",
            "Implement cross-validation (5-fold CV vs single train/test)",
            "Grid search for hyperparameters (K-Means k, tree depth)",
            "Add temporal features (how ratings change over time)",
        ],
        "Medium-term": [
            "Neural networks for preference prediction",
            "Temporal collaborative filtering (decay old ratings)",
            "Graph-based methods (social connections between users)",
            "Ensemble methods (combine multiple classifiers)",
            "SHAP values for model interpretability",
        ],
        "Long-term": [
            "Production deployment as REST API",
            "Real-time model retraining with new ratings",
            "Multi-objective optimization (accuracy vs diversity)",
            "Context-aware recommendations (time, social, mood)",
            "Fairness audits (check for recommendation bias)",
        ],
    }
    
    for phase, phase_items in recommendations.items():
        print(f"\n{phase} Enhancements:")
        for idx, item in enumerate(phase_items, 1):
            print(f"  {idx}. {item}")


print_future_recommendations()


# SECTION 11: FINAL SUMMARY

def print_final_summary():
    """Final summary and conclusions"""
    
    summary  """

PROJECT COMPLETION SUMMARY


✓ COMPLETED TASKS:
  • Data Loading & Integration: Combined movies.csv + ratings.csv
  • Data Cleaning: Fixed 7,963+ issues, achieved 100% data quality
  • Exploratory Analysis: Generated 4 histograms + genre analysis
  • Sampling: Tested random vs stratified (chose stratified)
  • Transformation: Applied log transform to reduce skewness
  • Normalization: Used MinMax and Z-Score scaling
  • Discretization: Created 4 rating categories + popularity ranks
  • PCA: Reduced 20 dims to 5 with 73% variance retention
  • Feature Selection: Identified top 5 predictive genres
  • SVD Compression: Tested 2-15 components, found optimal 5
  • Regression: Compared 3 models, best R²0.119
  • Clustering: Identified 8 natural movie clusters via K-Means
  • Content-Based: Built genre-similarity recommender
  • Collaborative: Built SVD-based user-pattern recommender
  • Hybrid: Combined both approaches with α0.6
  • Classification: Trained decision tree, 78.4% accuracy
  • Results Export: Saved all findings to Excel file

📊 MODELS BUILT:
  1. Linear/Multiple/Log-Linear Regression (prediction)
  2. K-Means Clustering (unsupervised grouping)
  3. Decision Tree Classifier (classification)
  4. Content-Based Filtering (recommendation)
  5. Collaborative Filtering (recommendation)
  6. Hybrid Recommender System (recommendation)

📈 KEY NUMBERS:
  • 16 distinct data mining techniques applied
  • 62,423 movies analyzed
  • 500,000 ratings processed (sample of 25M)
  • 20 genres tracked
  • 8 movie clusters identified
  • 78.4% classification accuracy achieved
  • 73% variance captured with 75% dimension reduction

💡 MAJOR FINDING:
  No single recommendation approach works best for all cases. Hybrid system
  combining content-based (explainability) and collaborative filtering
  (personalization) provides optimal balance for production systems.

🎓 LEARNING OUTCOMES:
  • Hands-on experience with full ML pipeline
  • Understanding of data quality importance
  • Practical knowledge of dimensionality reduction
  • Recommendation system design patterns
  • Model evaluation and interpretation skills
  • Real-world data handling (outliers, sparsity, noise)


"""
    
    print(summary)


print_final_summary()


# End of summary script
