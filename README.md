# Iris-Flower-Classification-using-K-Nearest-Neighbors-KNN
This project demonstrates a simple machine learning pipeline to classify Iris flower species using the K-Nearest Neighbors (KNN) algorithm. The model predicts the species of an iris flower based on its sepal and petal dimensions.

 Dataset Used
We use the built-in Iris dataset from sklearn.datasets, which includes:

Features:

Sepal Length (cm)

Sepal Width (cm)

Petal Length (cm)

Petal Width (cm)

Target (Species):

Setosa

Versicolor

Virginica

 Project Workflow
1️⃣ Import the Dataset
Load the Iris dataset using load_iris() from Scikit-learn.

Convert the dataset to a pandas DataFrame for easy viewing and analysis.

2️⃣ Explore the Data
View the first few rows using df.head().

Visualize feature relationships using sns.pairplot() and histograms.

3️⃣ Prepare the Data
Define X as features and y as target.

Split the data into training and testing sets using train_test_split().

4️⃣ Build the Model
Use the KNeighborsClassifier from Scikit-learn.

Set n_neighbors=3 for 3-nearest neighbors.

5️⃣ Train and Predict
Fit the model on training data.

Predict the species on the test data.

6️⃣ Evaluate the Model
Calculate the accuracy score of predictions.

Display a confusion matrix using seaborn.heatmap() to evaluate classification performance.

 Requirements
Install the required Python libraries using:

bash
Copy
Edit
pip install pandas scikit-learn matplotlib seaborn

Visualizations:

Pairplot to see feature relationships.

Confusion matrix to assess performance.

 File Structure
bash
Copy
Edit

iris-knn-classification/
├── iris_knn.py        # Main Python script
├── README.md          # Project documentation (this file)

 Conclusion
This beginner-friendly project is a great introduction to:

Scikit-learn's dataset and ML models

K-Nearest Neighbors algorithm

Data visualization with seaborn & matplotlib

Model evaluation metrics

 Author
Made by [J.JAWED AHAMED]
GitHub: Jawed2127
