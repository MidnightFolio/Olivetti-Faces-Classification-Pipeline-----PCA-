**Olivetti Faces Recognition & Dimensionality Reduction**

This is a computer vision and machine learning project that addresses the challenges of very high data dimensions by using Principal Component Analysis (PCA) for feature extraction, together with multi-model classification and a 5-fold cross-validation on the Olivetti Faces dataset.

---

**Project Overview**

Raw image data consists of massive pixel matrices. In this dataset, each facial image is 64 * 64 pixels, yielding 4,096 features per image. Training models directly on thousands of raw pixels is computationally heavy and prone to overfitting. 

Therefore, this project shows how to:
1. Visualize high-dimensional face datasets and extract Eigenfaces using PCA.
2. Compress features from 4,096 down to 100 principal components while retaining crucial variance.
3. Benchmark and compare three distinct machine learning classifiers (SVM, Logistic Regression, and Naive Bayes).
4. Evaluate model performance reliably using K-Fold Cross-Validation.

---

**Dataset & Features**

* **Dataset:** scikit-learn's built-in `fetch_olivetti_faces()` (40 distinct subjects, 10 images per subject).
* **Dimensionality Reduction Target:** 100 Principal Components.

---

**Tech Stack & Dependencies**

* **Python 3.x**
* **Scikit-Learn**: PCA, classifiers (`SVC`, `LogisticRegression`, `GaussianNB`), data splitting, and cross-validation (`KFold`).
* **Matplotlib**: Image grid plotting and PCA variance curve visualization.
* **NumPy**: Array manipulation and reshaping.

---

**Installation & Usage**

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/MidnightFolio/Olivetti-Faces-Classification-Pipeline-----PCA.git](https://github.com/MidnightFolio/Olivetti-Faces-Classification-Pipeline-----PCA.git)
   cd Olivetti-Faces-Classification-Pipeline-----PCA
2. **Install required packages:**
  ```bash

  pip install numpy scikit-learn matplotlib
```
3. **Run the script:**
  ```Bash
  python face_recognition.py
```
---
Methodology & Architecture

Exploratory Visualization: Plots raw 64 * 64 grayscale face matrices grouped by user ID.

Dimensionality Reduction (PCA): Fits Principal Component Analysis to the dataset to analyze cumulative explained variance, ensuring all necessary facial features are preserved within 100 components.

Eigenface Extraction: Reshapes PCA components back into 64 * 64 grids to visualize the structural eigenvectors (eigenfaces).

Multi-Model Benchmarking: Evaluates performance across Logistic Regression, Support Vector Machine (SVM), and Naive Bayes using a 5-fold cross-validation strategy.

---

Repository Structure

├── Face_Classification_Pipeline.py   # Main pipeline script (PCA + Classifiers + Visualization)

└── README.md             # Project documentation (This file)

---
**Author**

Thomas Oluwafemi Johnson

Agricultural & Bio-Resources Engineer | Data Science & Machine Learning Practitioner
