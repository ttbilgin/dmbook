# Comprehensive Data Mining Course Content

## 1. Introduction to Data Mining
### 1.1. What Is Data Mining?
1.2. Data Mining: An Essential Step in Knowledge Discovery
1.3. Motivating Challenges
1.4. The Origins of Data Mining
1.5. Data Mining Tasks and Applications
1.6. Diversity of Data Types for Data Mining
1.7. Mining Various Kinds of Knowledge
1.8. Data Mining: Confluence of Multiple Disciplines
1.9. Data Mining and Society

## 2. Data, Measurements, and Preprocessing
### 2.1. Types of Data
#### 2.1.1. Attributes and Measurement
#### 2.1.2. Types of Data Sets
2.2. Statistics of Data
2.3. Data Quality
   2.3.1. Measurement and Data Collection Issues
   2.3.2. Issues Related to Applications
2.4. Data Cleaning and Data Integration
2.5. Data Preprocessing
   2.5.1. Aggregation
   2.5.2. Sampling
   2.5.3. Dimensionality Reduction
   2.5.4. Feature Subset Selection
   2.5.5. Feature Creation
   2.5.6. Discretization and Binarization
   2.5.7. Variable Transformation
2.6. Measures of Similarity and Dissimilarity
   2.6.1. Basics of Proximity Measures
   2.6.2. Similarity and Dissimilarity between Simple Attributes
   2.6.3. Dissimilarities between Data Objects
   2.6.4. Similarities between Data Objects
   2.6.5. Examples of Proximity Measures
   2.6.6. Mutual Information
   2.6.7. Kernel Functions
   2.6.8. Bregman Divergence
   2.6.9. Issues in Proximity Calculation
   2.6.10. Selecting the Right Proximity Measure

## 3. Classification: Basic Concepts and Techniques
3.1. Basic Concepts
3.2. General Framework for Classification
3.3. Decision Tree Classifier
   3.3.1. A Basic Algorithm to Build a Decision Tree
   3.3.2. Methods for Expressing Attribute Test Conditions
   3.3.3. Measures for Selecting an Attribute Test Condition
   3.3.4. Algorithm for Decision Tree Induction
   3.3.5. Example Applications
   3.3.6. Characteristics of Decision Tree Classifiers
3.4. Model Overfitting
   3.4.1. Reasons for Model Overfitting
3.5. Model Selection
   3.5.1. Using a Validation Set
   3.5.2. Incorporating Model Complexity
   3.5.3. Estimating Statistical Bounds
   3.5.4. Model Selection for Decision Trees
3.6. Model Evaluation
   3.6.1. Holdout Method
   3.6.2. Cross-Validation
3.7. Presence of Hyper-parameters
   3.7.1. Hyper-parameter Selection
   3.7.2. Nested Cross-Validation
3.8. Pitfalls of Model Selection and Evaluation
3.9. Model Comparison

## 4. Classification: Alternative Techniques
4.1. Types of Classifiers
4.2. Rule-Based Classifier
   4.2.1. How a Rule-Based Classifier Works
   4.2.2. Properties of a Rule Set
   4.2.3. Direct Methods for Rule Extraction
   4.2.4. Indirect Methods for Rule Extraction
   4.2.5. Characteristics of Rule-Based Classifiers
4.3. Nearest Neighbor Classifiers
   4.3.1. Algorithm
   4.3.2. Characteristics of Nearest Neighbor Classifiers
4.4. Naïve Bayes Classifier
   4.4.1. Basics of Probability Theory
   4.4.2. Naïve Bayes Assumption
4.5. Bayesian Networks
   4.5.1. Graphical Representation
   4.5.2. Inference and Learning
   4.5.3. Characteristics of Bayesian Networks
4.6. Logistic Regression
   4.6.1. Logistic Regression as a Generalized Linear Model
   4.6.2. Learning Model Parameters
   4.6.3. Characteristics of Logistic Regression
4.7. Artificial Neural Network (ANN)
   4.7.1. Perceptron
   4.7.2. Multi-layer Neural Network
   4.7.3. Characteristics of ANN
4.8. Support Vector Machine (SVM)
   4.8.1. Margin of a Separating Hyperplane
   4.8.2. Linear SVM
   4.8.3. Soft-margin SVM
   4.8.4. Nonlinear SVM
   4.8.5. Characteristics of SVM

## 5. Deep Learning
5.1. Basic Concepts
5.2. Deep Learning Techniques
   5.2.1. Using Synergistic Loss Functions
   5.2.2. Using Responsive Activation Functions
   5.2.3. Regularization
   5.2.4. Initialization of Model Parameters
   5.2.5. Characteristics of Deep Learning
5.3. Improve Training of Deep Learning Models
5.4. Convolutional Neural Networks
5.5. Recurrent Neural Networks
5.6. Graph Neural Networks

## 6. Ensemble Methods and Classification Challenges
6.1. Ensemble Methods
   6.1.1. Rationale for Ensemble Method
   6.1.2. Methods for Constructing an Ensemble Classifier
   6.1.3. Bias-Variance Decomposition
   6.1.4. Bagging
   6.1.5. Boosting
   6.1.6. Random Forests
   6.1.7. Empirical Comparison among Ensemble Methods
6.2. Class Imbalance Problem
   6.2.1. Building Classifiers with Class Imbalance
   6.2.2. Evaluating Performance with Class Imbalance
   6.2.3. Finding an Optimal Score Threshold
   6.2.4. Aggregate Evaluation of Performance
6.3. Multiclass Problem

## 7. Pattern Mining: Basic Concepts and Methods
7.1. Basic Concepts
7.2. Frequent Itemset Mining Methods
   7.2.1. The Apriori Principle
   7.2.2. Frequent Itemset Generation in the Apriori Algorithm
   7.2.3. Candidate Generation and Pruning
   7.2.4. Support Counting
   7.2.5. Computational Complexity
7.3. Rule Generation
   7.3.1. Confidence-Based Pruning
   7.3.2. Rule Generation in Apriori Algorithm
   7.3.3. Example Applications
7.4. Which Patterns are Interesting?—Pattern Evaluation Methods
   7.4.1. Objective Measures of Interestingness
   7.4.2. Measures beyond Pairs of Binary Variables
   7.4.3. Simpson's Paradox
7.5. Effect of Skewed Support Distribution

## 8. Pattern Mining: Advanced Methods
8.1. Mining Various Kinds of Patterns
8.2. Mining Compressed or Approximate Patterns
   8.2.1. Maximal Frequent Itemsets
   8.2.2. Closed Itemsets
8.3. Constraint-Based Pattern Mining
8.4. Alternative Methods for Generating Frequent Itemsets
8.5. FP-Growth Algorithm
   8.5.1. FP-Tree Representation
   8.5.2. Frequent Itemset Generation in FP-Growth Algorithm
8.6. Handling Categorical Attributes
8.7. Handling Continuous Attributes
   8.7.1. Discretization-Based Methods
   8.7.2. Statistics-Based Methods
   8.7.3. Non-discretization Methods
8.8. Handling a Concept Hierarchy
8.9. Sequential Patterns
   8.9.1. Preliminaries
   8.9.2. Sequential Pattern Discovery
   8.9.3. Timing Constraints
   8.9.4. Alternative Counting Schemes
8.10. Mining Subgraph Patterns
    8.10.1. Preliminaries
    8.10.2. Frequent Subgraph Mining
    8.10.3. Candidate Generation
    8.10.4. Candidate Pruning
    8.10.5. Support Counting
8.11. Infrequent Patterns
    8.11.1. Negative Patterns
    8.11.2. Negatively Correlated Patterns
    8.11.3. Comparisons among Pattern Types
    8.11.4. Techniques for Mining Interesting Infrequent Patterns
    8.11.5. Techniques Based on Mining Negative Patterns
    8.11.6. Techniques Based on Support Expectation
8.12. Pattern Mining: Application Examples

## 9. Cluster Analysis: Basic Concepts and Methods
9.1. Cluster Analysis Overview
    9.1.1. What Is Cluster Analysis?
    9.1.2. Different Types of Clusterings
    9.1.3. Different Types of Clusters
9.2. Partitioning Methods
    9.2.1. The Basic K-means Algorithm
    9.2.2. K-means: Additional Issues
    9.2.3. Bisecting K-means
    9.2.4. K-means and Different Types of Clusters
    9.2.5. Strengths and Weaknesses
    9.2.6. K-means as an Optimization Problem
9.3. Hierarchical Methods
    9.3.1. Basic Agglomerative Hierarchical Clustering Algorithm
    9.3.2. Specific Techniques
    9.3.3. The Lance-Williams Formula for Cluster Proximity
    9.3.4. Key Issues in Hierarchical Clustering
    9.3.5. Outliers
    9.3.6. Strengths and Weaknesses
9.4. Density-Based and Grid-Based Methods
    9.4.1. Traditional Density: Center-Based Approach
    9.4.2. The DBSCAN Algorithm
    9.4.3. Strengths and Weaknesses
9.5. Evaluation of Clustering
    9.5.1. Overview
    9.5.2. Unsupervised Cluster Evaluation Using Cohesion and Separation
    9.5.3. Unsupervised Cluster Evaluation Using the Proximity Matrix
    9.5.4. Unsupervised Evaluation of Hierarchical Clustering
    9.5.5. Determining the Correct Number of Clusters
    9.5.6. Clustering Tendency
    9.5.7. Supervised Measures of Cluster Validity
    9.5.8. Assessing the Significance of Cluster Validity Measures
    9.5.9. Choosing a Cluster Validity Measure

## 10. Cluster Analysis: Advanced Methods
10.1. Characteristics of Data, Clusters, and Clustering Algorithms
    10.1.1. Data Characteristics
    10.1.2. Cluster Characteristics
    10.1.3. General Characteristics of Clustering Algorithms
10.2. Probabilistic Model-Based Clustering
    10.2.1. Fuzzy Clustering
    10.2.2. Clustering Using Mixture Models
    10.2.3. Self-Organizing Maps (SOM)
10.3. Clustering High-Dimensional Data
10.4. Biclustering
10.5. Dimensionality Reduction for Clustering
10.6. Clustering Graph and Network Data
    10.6.1. Graph-Based Clustering
    10.6.2. Sparsification
    10.6.3. Minimum Spanning Tree (MST) Clustering
    10.6.4. OPOSSUM: Optimal Partitioning of Sparse Similarities Using METIS
    10.6.5. Chameleon: Hierarchical Clustering with Dynamic Modeling
    10.6.6. Spectral Clustering
    10.6.7. Shared Nearest Neighbor Similarity
    10.6.8. The Jarvis-Patrick Clustering Algorithm
    10.6.9. SNN Density
    10.6.10. SNN Density-Based Clustering
10.7. Scalable Clustering Algorithms
    10.7.1. Scalability: General Issues and Approaches
    10.7.2. BIRCH
    10.7.3. CURE
10.8. Semisupervised Clustering
10.9. Which Clustering Algorithm?

## 11. Outlier Detection
11.1. Basic Concepts
    11.1.1. A Definition of an Anomaly
    11.1.2. Nature of Data
    11.1.3. How Anomaly Detection is Used
11.2. Characteristics of Anomaly Detection Methods
11.3. Statistical Approaches
    11.3.1. Using Parametric Models
    11.3.2. Using Non-parametric Models
    11.3.3. Modeling Normal and Anomalous Classes
    11.3.4. Assessing Statistical Significance
    11.3.5. Strengths and Weaknesses
11.4. Proximity-Based Approaches
    11.4.1. Distance-based Anomaly Score
    11.4.2. Density-based Anomaly Score
    11.4.3. Relative Density-based Anomaly Score
    11.4.4. Strengths and Weaknesses
11.5. Clustering- vs. Classification-Based Approaches
    11.5.1. Finding Anomalous Clusters
    11.5.2. Finding Anomalous Instances
    11.5.3. Strengths and Weaknesses
11.6. Reconstruction-Based Approaches
    11.6.1. Strengths and Weaknesses
11.7. Mining Contextual and Collective Outliers
11.8. Outlier Detection in High-Dimensional Data
11.9. Evaluation of Anomaly Detection
