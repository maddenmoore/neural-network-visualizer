Neural Network Visualizer - 15-112 Term Project by Madden Moore

Description:
My term project involves a neural network made with no external machine learning modules for the purpose of classifying breast tumors as benign or malignant
The UI for this project allows the user to customize input paramaters, see how well the model can classify the data with their input paramaters, and track results over multiple model runs
The UI also provides a shallow explanation of how a neural network works

How to run:
The user should run "userinterface.py" which will create a popup which is the source of the function
The user should have "brca_tumor_data.csv", "neuralnetwork.py", and "accuracies.txt" in the same folder as "userinterface.py"
Optional: "sampleData.txt" includes a set of results I have created over many model runs. To load this file, it must be placed in the same folder as "userinterface.py"


Libraries needed:
Pandas (https://pandas.pydata.org/)
Numpy (https://numpy.org/)
Matplotlib (https://matplotlib.org/)

Shortcut commands:
On the explanation screen (press 'E' on the start screen), pressing '0' takes you straight to the historical results screen instead of having to wait for the completion of the model

Other notes:
Included in the submission folder are several .png files showcasing graphs or result screens from various model runs
These help provide quick visualizations of model improvement over time


Data source:
Data was obtained from https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29 and edited by myself to only include relevant data columns
