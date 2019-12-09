#neuralnetwork is from my neuralnetwork.py file
import neuralnetwork as network
#cmu_112_graphics obtained from https://www.cs.cmu.edu/~112/notes/cmu_112_graphics.py
#modification made on line 272 to allow showMessage to work with a Modal App
from cmu_112_graphics_modified import *
from tkinter import *
#pandas obtained from https://pandas.pydata.org/
import pandas as pd
#numpy obtained from https://numpy.org/
import numpy as np
#matplotlib obtained from https://matplotlib.org/
import matplotlib.pyplot as plt
import os
import string

#from https://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-float
#tries to convert the string to a flot, returns true if it can, returns false otherwise
def isDecimal(val):
    if val == "NaN" or val == "nan": return False
    try:
        float(val)
        return True
    except:
        return False

#adds a row of the output data to the dataFrame
def addRowToAccuracies(data, newTrainPercent, newLR, newNumNeurons, newAccuracy, newCost):
    data2 = pd.DataFrame({"Train Percent": [newTrainPercent], "Learning Rate": [newLR],
    "Number of Neurons": [newNumNeurons], "Test Accuracy": [newAccuracy], 
    "Test Cost": [newCost]})
    return data.append(data2)

#clears the current accuracy data frame
def clearAccuracies():
    blank = pd.DataFrame()
    blank.to_csv("accuracies.txt")

#reads in the current list of accuracies 
def csvToDataFrame(path = "accuracies.txt"):
    #the size of a blank dataframe is 4
    if os.path.getsize(path) == 4 or os.path.getsize(path) == 0:
        
        return pd.DataFrame()
    data = pd.read_csv(path, sep=",")
    return data


#subclasses the modal app class from cmu_112_graphics.py
class NeuralNetworkVisualizer(ModalApp):
    def appStarted(app):
        app.startScreen = StartScreen()
        app.networkDrawing = NetworkDrawing()
        app.inputScreen = InputScreen()
        app.featureScreen = FeatureScreen()
        app.explanationScreen = ExplanationScreen()
        app.historyScreen = HistoryScreen()
        app.setActiveMode(app.startScreen)

#subclasses the Mode class from cmu_112_graphics.py
class ExplanationScreen(Mode):

    #from cmu_112_graphics.py
    def redrawAll(mode, canvas):
        text = "Each layer consists of neurons, which are objects that store a numerical value"
        canvas.create_text(mode.width/2, 50, text = text, font = "Times 16")
        text = "Each neuron is connected to every neuron in the next layer, this connection is a value called a weight"
        canvas.create_text(mode.width/2, 100, text = text, font = "Times 15")
        text = "The value of each hidden layer and output neuron is as follows: "
        canvas.create_text(mode.width/2, 150, text = text, font = "Times 16")
        text = "Value = sum of previous layer's neurons * their respective weights plus a constant (bias)"
        canvas.create_text(mode.width/2, 200, text = text, font = "Times 16")
        text = "The model 'learns' by making predictions on training data it already knows the answers to"
        canvas.create_text(mode.width/2, 250, text = text, font = "Times 15")
        text = "The learning happens by adjusting the weights and biases to minimize prediction error"
        canvas.create_text(mode.width/2, 300, text = text, font = "Times 15")
        text = "This process repeats until the cost function, which measures error, stops decreasing"
        canvas.create_text(mode.width/2, 350, text = text, font = "Times 16")
        text = "This process is called gradient descent optimization"
        canvas.create_text(mode.width/2, 400, text = text, font = "Times 16")
        text = "After the model is trained, it makes predictions on data it has never seen before"
        canvas.create_text(mode.width/2, 450, text = text, font = "Times 16")
        text = "The measure of model performance is how well it performs on this data"
        canvas.create_text(mode.width/2, 500, text = text, font = "Times 16")
        text = "Press 'R' to return to the start screen"
        canvas.create_text(mode.width/2, 550, text = text, font = "Times 16")

    #from cmu_112_graphics.py
    def keyPressed(mode, event):
        if event.key == "R":
            mode.app.setActiveMode(mode.app.startScreen)
        elif event.key == "0":
            mode.app.setActiveMode(mode.app.historyScreen)

#subclasses the Mode class from cmu_112_graphics.py
class FeatureScreen(Mode):

    #from cmu_112_graphics.py
    def redrawAll(mode, canvas):
        text = "The features used for this network are as follows:"
        canvas.create_text(mode.width/2, 100, text = text, font = "Times 16")
        text = "Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, "
        canvas.create_text(mode.width/2, 200, text = text, font = "Times 16")
        text = "Concave Points, Symmetry, and Fractal Dimension"
        canvas.create_text(mode.width/2, 300, text = text, font = "Times 16")
        text = "Press 'R' to return to the start screen"
        canvas.create_text(mode.width/2, 400, text = text, font = "Times 16")
    
    #from cmu_112_graphics.py
    def keyPressed(mode, event):
        if event.key == "R":
            mode.app.setActiveMode(mode.app.startScreen)

#subclasses the mode class from cmu_112_graphics.py
class InputScreen(Mode):
    #from cmu_112_graphics.py
    def appStarted(mode):
        mode.trainingPercent = 0
        mode.learningRate = 0
        mode.numNeurons = 0
        mode.testAccuracy = "None"
        mode.testCost = "None"
        mode.canStart = False
        mode.started = False
        mode.finished = False
        mode.training = False
        mode.data = csvToDataFrame()
        
    
    #inspired by http://www.cs.cmu.edu/~112/notes/notes-animations-part2.html#ioMethods
    def getNumNeurons(mode):
        oldNumNeurons = mode.numNeurons
        text = "Enter the number of hidden layer neurons (integer between 10 and 40, recommended: 20)"
        mode.numNeurons = mode.getUserInput(text)
        #if the input is not an integer between 10 and 40, get another input
        if (mode.numNeurons == None or not mode.numNeurons.isdigit() or
        int(mode.numNeurons) < 10 or int(mode.numNeurons) > 40):
            mode.numNeurons = oldNumNeurons
            mode.getNumNeurons()
        else:
            mode.numNeurons = int(mode.numNeurons)
    
    #inspired by http://www.cs.cmu.edu/~112/notes/notes-animations-part2.html#ioMethods
    def getTrainPercent(mode):
        oldTrainingPercent = mode.trainingPercent
        text = "Enter the percent of the data (integer < 100, recommended: 80) you want to train the model on"
        mode.trainingPercent = mode.getUserInput(text)
        #if the input is not an integer between 1 and 100, get another input
        if mode.trainingPercent == None or not mode.trainingPercent.isdigit() or int(mode.trainingPercent) >= 100:
            mode.trainingPercent = oldTrainingPercent
            mode.getTrainPercent()
        else:
            mode.trainingPercent = int(mode.trainingPercent)
    
    #inspired by http://www.cs.cmu.edu/~112/notes/notes-animations-part2.html#ioMethods
    def getLearningRate(mode):
        oldLearningRate = mode.learningRate
        text = "Enter the learning rate of the model (decimal between 0 and 1, not inclusive, recommended: 0.5)"
        mode.learningRate = mode.getUserInput(text)
        #if the input is not a decimal between 0 and 1, get another input
        if (mode.learningRate == None or not isDecimal(mode.learningRate) or
        float(mode.learningRate) <= 0 or float(mode.learningRate) > 1):
            mode.learningRate = oldLearningRate
            mode.getLearningRate()
        else:
            mode.learningRate = float(mode.learningRate)

    #from cmu_112_graphics.py
    def timerFired(mode):
        #the network cna start if all the input paramaters have been filled
        if mode.numNeurons > 0 and mode.learningRate > 0 and mode.trainingPercent > 0:
            mode.canStart = True
                

    #from cmu_112_graphics.py
    def redrawAll(mode, canvas):
        #if the model has not started, the mode gets the input data from the user
        if not mode.started:
            text = "Press 'T' to enter your training percent"
            canvas.create_text(mode.width/2, 50, text = text, font = "Times 16")
            text = f"Your training percent: {mode.trainingPercent}"
            canvas.create_text(mode.width/2, 100, text = text, font = "Times 16")
            text = "Press 'L' to enter your learning rate"
            canvas.create_text(mode.width/2, 150, text = text, font = "Times 16")
            text = f"Your learning rate: {mode.learningRate}"
            canvas.create_text(mode.width/2, 200, text = text, font = "Times 16")
            text = "Press 'N' to enter your number of neurons"
            canvas.create_text(mode.width/2, 250, text = text, font = "Times 16")
            text = f"Your number of neurons: {mode.numNeurons}"
            canvas.create_text(mode.width/2, 300, text = text, font = "Times 16")
            text = "The learning rate is the magnitude by which the model adjusts its weights"
            canvas.create_text(mode.width/2, 350, text = text, font = "Times 16")
            text = "The number of neurons determines the number of neurons in the hidden layer"
            canvas.create_text(mode.width/2, 400, text = text, font = "Times 16")
            if mode.canStart:
                text = "Press 'S' to start the model. The mode will automatically change screens when training is complete"
                canvas.create_text(mode.width/2, 450, text = text, font = "Times 15")
            text = "Press 'R' to return to the start screen"
            canvas.create_text(mode.width/2, 500, text = text, font = "Times 16")
        else:
            #attempting to display the current paramaters while the model is running
            if not mode.finished:
                text = f"Training model with {mode.trainingPercent}% training, {mode.learningRate}"
                text += f" learning rate, and {mode.numNeurons} neurons"
                canvas.create_text(mode.width/2, 50, text = text, font = "Times 16")
                text = f"Current Epoch (iteration): {len(mode.network.accuracies)}"
                canvas.create_text(mode.width/2, 100, text = text, font = "Times 16")
                if len(mode.network.accuracies) == 0:
                    text = "Last accuracy: None\nLast cost: None"
                    canvas.create_text(mode.width/2, 150, text = text, font = "Times 16")
                else:
                    text = f"First accuracy: {mode.network.accuracies[0]}"
                    text += f"\nFirst cost: {mode.network.costs[0]}"
                    canvas.create_text(mode.width/2, 150, text = text, font = "Times 16")
                    if len(mode.accuracies) > 1:
                        text = "Most recent accuracy: %.4f" % mode.network.accuracies[-1]
                        text += "\nMost recent cost: %.4f" % mode.network.costs[-1]
            else:
                #when the model is finished, display the final paramaters
                text = "Training complete. Press 'T' to test the model on testing data"
                canvas.create_text(mode.width/2, 50, text = text, font = "Times 16")
                text = "Initial training accuracy: %.4f" % mode.network.accuracies[0]
                canvas.create_text(mode.width/2, 100, text = text, font = "Times 16")
                text = "Final training accuracy: %.4f" % mode.network.accuracies[-1]
                canvas.create_text(mode.width/2, 150, text = text, font = "Times 16")
                text = "Initial training cost: %.4f" % mode.network.costs[0]
                canvas.create_text(mode.width/2, 200, text = text, font = "Times 16")
                text = "Final training cost: %.4f" % mode.network.costs[-1]
                canvas.create_text(mode.width/2, 250, text = text, font = "Times 16")
                if not isinstance(mode.testAccuracy, str):
                    text = "Test accuracy: %.4f" % mode.testAccuracy
                    canvas.create_text(mode.width/2, 300, text = text, font = "Times 16")
                    text = "Text cost: %.4f" % mode.testCost
                    canvas.create_text(mode.width/2, 350, text = text, font = "Times 16")
                text = "Press 'A' to view a plot of training accuracy"
                canvas.create_text(mode.width/2, 400, text = text, font = "Times 16")
                text = "Press 'C' to view a plot of cost over time"
                canvas.create_text(mode.width/2, 450, text = text, font = "Times 16")
                text = "Press 'R' to return to the start screen"
                canvas.create_text(mode.width/2, 500, text = text, font = "Times 16")
                text = "Press 'H' to view and graph historical run data"
                canvas.create_text(mode.width/2, 550, text = text, font = "Times 16")
                text = "Press 'S' to input different paramaters"
                canvas.create_text(mode.width/2, 600, text = text, font = "Times 16")
                


                    

    #from cmu_112_graphics.py
    def keyPressed(mode, event):
        if not mode.finished:
            if not mode.started:
                if event.key == "R":
                    mode.app.setActiveMode(mode.app.startScreen)
                elif event.key == "T":
                    mode.getTrainPercent()
                elif event.key == "L":
                    mode.getLearningRate()
                elif event.key == "N":
                    mode.getNumNeurons()
                elif event.key == "S" and mode.canStart:
                    mode.started = True
                    mode.training = True
                    data = network.preprocessData()
                    train, mode.testData, trainDiagnosis, mode.testDiagnosis = network.partitionTrainAndTest(data,
                    mode.trainingPercent)
                    mode.network = network.NeuralNetwork(train, trainDiagnosis, 
                    mode.numNeurons, mode.learningRate)
                    (mode.finished, mode.accuracies, mode.costs) = mode.network.train()
                    mode.epochs = range(len(mode.accuracies))
        else:
            mode.data = csvToDataFrame()
            #plot variables over time
            if event.key == "A":
                plt.clf()
                plt.title("Training Accuracy Over Time")
                plt.xlabel("Epoch")
                plt.ylabel("Training Accuracy")
                plt.plot(mode.epochs, mode.accuracies)
                plt.show()
            elif event.key == "C":
                plt.clf()
                plt.title("Cost over time")
                plt.xlabel("Epoch")
                plt.ylabel("Cost")
                plt.plot(mode.epochs, mode.costs)
                plt.show()
            elif event.key == "R":
                plt.clf()
                mode.app.inputScreen.appStarted()
                mode.app.setActiveMode(mode.app.startScreen)
            elif event.key == "T":
                if isinstance(mode.testAccuracy, str):
                    mode.testAccuracy, mode.testCost = mode.network.test(mode.testData, mode.testDiagnosis)
                    mode.data = addRowToAccuracies(mode.data, mode.trainingPercent,
                    mode.learningRate, mode.numNeurons, mode.testAccuracy,
                    mode.testCost)
                    mode.data.to_csv("accuracies.txt")
            elif event.key == "H":
                mode.app.setActiveMode(mode.app.historyScreen)
            elif event.key == "S":
                mode.appStarted()
                
                
#subclasses the mode class from cmu_112_graphics.py
class HistoryScreen(Mode):

    def appStarted(mode):
        mode.data = csvToDataFrame()

    def redrawAll(mode, canvas):
        text = "Press '1' to see a plot of training percent vs test accuracy over multiple runs"
        canvas.create_text(mode.width/2, 50, text = text, font = "Times 16")
        text = "Press '2' to see a plot of learning rate vs test accuracy over multiple runs"
        canvas.create_text(mode.width/2, 100, text = text, font = "Times 16")
        text = "Press '3' to see a plot of number of neurons vs test accuracy over multiple runs"
        canvas.create_text(mode.width/2, 150, text = text, font = "Times 16")
        text = "Press 'X' to clear the history of training accuracies"
        canvas.create_text(mode.width/2, 200, text = text, font = "Times 16")
        text = "Press 'S' to save the current dataset of training accuracies"
        canvas.create_text(mode.width/2, 250, text = text, font = "Times 16")
        text = "Press 'L' to load previously saved results"
        canvas.create_text(mode.width/2, 300, text = text, font = "Times 16")
        text = "Press 'I' to return to the input screen"
        canvas.create_text(mode.width/2, 350, text = text, font = "Times 16")
        text = "Press 'R' to return to the start screen"
        canvas.create_text(mode.width/2, 400, text = text, font = "Times 16")


    def keyPressed(mode, event):
        mode.data = csvToDataFrame()
        if event.key == "I":
            mode.app.inputScreen.appStarted()
            mode.app.setActiveMode(mode.app.inputScreen)
        elif event.key == "R":
            mode.app.setActiveMode(mode.app.startScreen)
        #graph variables vs test accuracy over multiple runs
        elif event.key == "1":
            if len(mode.data) == 0:
                mode.showMessage("There is no test history!")
            else:
                plt.clf()
                plt.title("Training Percent vs Test Accuracy")
                plt.xlabel("Training Percent")
                plt.ylabel("Test Accuracy")
                plt.scatter(mode.data["Train Percent"], mode.data["Test Accuracy"])
                plt.show()
        elif event.key == "2":
            if len(mode.data) == 0:
                mode.showMessage("There is no test history!")
            else:
                plt.clf()
                plt.title("Learning Rate vs Test Accuracy")
                plt.xlabel("Learning Rate")
                plt.ylabel("Test Accuracy")
                plt.scatter(mode.data["Learning Rate"], mode.data["Test Accuracy"])
                plt.show()
        elif event.key == "3":
            if len(mode.data) == 0:
                mode.showMessage("There is no test history!")
            else:
                plt.clf()
                plt.title("Number of Neurons vs Test Accuracy")
                plt.xlabel("Number of Neurons")
                plt.ylabel("Test Accuracy")
                plt.scatter(mode.data["Number of Neurons"], mode.data["Test Accuracy"])
                plt.show()
        #clear data upon user input
        elif event.key == "X":
            answer = mode.getUserInput("Are you sure you want to clear your results? Input 'yes' to clear (this cannot be undone)")
            if isinstance(answer, str) and answer.lower() == "yes":
                clearAccuracies()
                mode.showMessage("Results cleared!")
        elif event.key == "S":
            #get file input from user, check to see if the file already exists
            #if the file does not exist, just create it
            #otherwise, ask the user if they want to overwrite the file
            answer = mode.getUserInput("Enter the name of the file you want to load (.txt file)")
            if isinstance(answer, str):
                if not answer.endswith(".txt"):
                    mode.showMessage(f"{answer} is not a .txt file!")
                else:
                    try:
                        os.path.getsize(answer)
                        response = mode.getUserInput("Are you sure you want to overwrite this file? This cannot be undone. Type 'yes' to continue")
                        if isinstance(response, str) and response.lower() == "yes":
                            mode.data.to_csv(answer)
                            mode.showMessage("Data saved as " + answer)
                        else:
                            mode.showMessage("Didn't save to file " + answer)
                    except:
                        mode.data.to_csv(answer)
                        mode.showMessage("Data saved as " + answer)
        elif event.key == "L":
            #get a filename, see if it exists
            #if it does exist, ask the user if they want to overwrite their current results
            #if the file is a blank dataframe (size == 4), don't load it
            answer = mode.getUserInput("Enter the name of the file you want to load (.txt file)")
            if isinstance(answer, str):
                if not answer.endswith(".txt"):
                    mode.showMessage("This file is not a .txt file!")
                else:
                    try:
                        os.path.getsize(answer)
                        response = mode.getUserInput("Are you sure you want to load this file? This will overwrite the current results. Type 'yes' to continue")
                        if isinstance(response, str) and response.lower() == "yes":
                            if os.path.getsize(answer) != 0 and os.path.getsize(answer) != 4:
                                mode.data = csvToDataFrame(answer)
                                mode.data.to_csv("accuracies.txt")
                                mode.showMessage(f"File {answer} loaded")
                            else:
                                mode.showMessage("The file you entered is blank!")
                        else:
                            mode.showMessage("Didn't load file " + answer)
                    except:
                        mode.showMessage(f"File {answer} does not exist!")


#subclasses the mode class from cmu_112_graphics.py
class NetworkDrawing(Mode):
    
    #from cmu_112_graphics.py
    def appStarted(mode):
        mode.inputX = 200
        mode.hiddenX = 400
        mode.outputX = 600
        mode.radius = 50
    
    #from cmu_112_graphics.py
    def redrawAll(mode, canvas):
        mode.drawInputLayer(canvas)
        mode.drawHiddenLayer(canvas)
        mode.drawOutputLayer(canvas)
        text = "Each circle represents one neuron, each line represents connections between neurons"
        canvas.create_text(mode.width/2, 30, text = text, font = "Times 16")
        text = "The value of the output neuron is the model's prediction"
        canvas.create_text(300, 75, text = text, font = "Times 16", anchor = "w")
        canvas.create_text(200, 775, text = "Input Layer", font = "Times 16")
        canvas.create_text(400, 775, text = "Hidden Layer", font = "Times 16")
        canvas.create_text(600, 775, text = "Output Layer", font = "Times 16")
        canvas.create_text(mode.outputX, 400, text = "0.82", font = "Times 16")
        text = ">= 0.5: Malignant\n<0.5: Benign"
        canvas.create_text(mode.outputX, 500, text = text, font = "Times 16")
        text = "Press 'R' to return to the start screen"
        canvas.create_text(mode.outputX, 650, text = text, font = "Times 16")

    #from cmu_112_graphics.py
    def keyPressed(mode, event):
        if event.key == "R":
            mode.app.setActiveMode(mode.app.startScreen)

    def drawHiddenLayer(mode, canvas):
        startY = 200
        endY = 600
        outputY = 400
        outputLineX = mode.outputX - mode.radius
        for i in range(startY, endY+1, 200):
            canvas.create_oval(mode.hiddenX - mode.radius, i - mode.radius,
            mode.hiddenX + mode.radius, i + mode.radius)
            canvas.create_line(mode.hiddenX + mode.radius, i, outputLineX, outputY)
    
    def drawOutputLayer(mode, canvas):
        y = 400
        canvas.create_oval(mode.outputX - mode.radius, y - mode.radius,
        mode.outputX + mode.radius, y + mode.radius)

    
    def drawInputLayer(mode, canvas):
        startY = 100
        endY = 700
        hiddenLineX = mode.hiddenX - mode.radius
        hiddenLineY = [200, 400, 600]
        for i in range(startY, endY+1, 200):
            canvas.create_oval(mode.inputX - mode.radius, i - mode.radius,
            mode.inputX + mode.radius, i + mode.radius)
            for y in hiddenLineY:
                canvas.create_line(mode.inputX + mode.radius, i, hiddenLineX, y)



#subclasses the mode class from cmu_112_graphics.py
#functions redrawAll and keyPressed are from this package 
class StartScreen(Mode):

    #from cmu_112_graphics
    def redrawAll(mode, canvas):
        canvas.create_text(mode.width/2, 10, text = "Welcome to the Tumor Classifying Neural Network",
        anchor = "n", font = "Times 20")
        text = "This neural network helps to classify tumors as benign or malignant"
        text += " based on physical features"
        canvas.create_text(mode.width/2, 100, text = text, font = "Times 16")
        text = "Training a neural network for this task could lead to more accurate"
        text += " medical diagnosis"
        canvas.create_text(mode.width/2, 200, text = text, font = "Times 16")
        text = "To see a list of physical features, press 'F'"
        canvas.create_text(mode.width/2, 300, text = text, font = "Times 16")
        text = "To view a representation of a neural network, press 'N'"
        canvas.create_text(mode.width/2, 400, text = text, font = "Times 16")
        text = "To view an explanation of the neural network, press 'E'"
        canvas.create_text(mode.width/2, 500, text = text, font = "Times 16")
        text = "To start the model, press 'S'"
        canvas.create_text(mode.width/2, 600, text = text, font = "Times 16")

    #from cmu_112_graphics
    def keyPressed(mode, event):
        if event.key == "F":
            mode.app.setActiveMode(mode.app.featureScreen)
        elif event.key == "N":
            mode.app.setActiveMode(mode.app.networkDrawing)
        elif event.key == "S":
            mode.app.inputScreen.appStarted()
            mode.app.setActiveMode(mode.app.inputScreen)
        elif event.key == "E":
            mode.app.setActiveMode(mode.app.explanationScreen)
    
    

def run():
    NeuralNetworkVisualizer(width = 800, height = 800)

if (__name__ == '__main__'):
    run()