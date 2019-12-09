#hand coded neural network
#model runs can vary from trial to trial due to the random intitialization of weights

#pandas obtained from https://pandas.pydata.org/
import pandas as pd
#numpy obtained from https://numpy.org/
import numpy as np
import random
import math
import decimal
import copy

#reads in the data and replaces the designations for "benign" and "malignant"
#("B" and "M" in the orginal file) with 0 and 1 respectively
def preprocessData():
    data = pd.read_csv("brca_tumor_data.csv", header = 0, index_col = 0)
    data = data.replace("B", 0)
    data = data.replace("M", 1)
    data = scaleData(data)
    return data

#scale the data so that each value is proportional to its own percentage
#of the total sum of that value for the dataset
#prevents any one data point from having more input on the training than any other
def scaleData(data):
    result = list()
    numRows = len(data)
    colNames = data.columns.tolist()
    for i in range(numRows):
        row = [list(data["diagnosis"])[i]]
        #exclude "diagnosis" column - no need to scale it since it's already 1s and 0s
        for j in range(1, 11):
            row.append(list(data[colNames[j]])[i]/sum(data[colNames[j]])*data.shape[0])
        result.append(row)
    data2 = pd.DataFrame(result, columns = colNames)
    return data2


#splits the data into the training and testing datasets based off of the 
#trainSplit variable. No tumor exists in both datasets
def partitionTrainAndTest(data, trainSplit):
    assert(type(trainSplit) == int)
    numRows = len(data)
    numRowsToChoose = int(numRows * trainSplit / 100)
    testSplit = 100 - trainSplit
    trainData = pd.DataFrame()
    testData = pd.DataFrame()
    #create list of indicies which will be the training data
    #convert to a set for efficiency
    resultList = set(random.sample(list(range(numRows)), numRowsToChoose))
    for i in range(numRows):
        if i in resultList:
            trainData = trainData.append(data.iloc[i])
        else:
            testData = testData.append(data.iloc[i])
    #split the training and testing data so that the expected
    #results are in one dataframe and the variables are in the other
    diagnosisIndex = 4
    colNames = trainData.columns.tolist()
    dataCols = colNames[:4] + colNames[5:]
    diagnosisCols = [colNames[diagnosisIndex]]
    trainDiagnosis = trainData[diagnosisCols]
    trainData = trainData[dataCols]
    testDiagnosis = testData[diagnosisCols]
    testData = testData[dataCols]
    return trainData, testData, trainDiagnosis, testDiagnosis

#placeholder run function if the just this python file is used
#this function is mainly for debugging purposes
def run(trainSplt = 80, numNeurons2 = 20, learningRate = 0.4):
    #can set the random seed the same every run for debugging purposes
    #random.seed(42) 
    #np.random.seed(4243)
    print("Preprocessing the data...")
    data = preprocessData()
    trainSplit = trainSplt
    train, test, trainDiagnosis, testDiagnosis = partitionTrainAndTest(data,
    trainSplit)
    numNeurons = numNeurons2
    network = NeuralNetwork(train, trainDiagnosis, numNeurons, learningRate)
    print("Training network...")
    network.train()
    #input("Training complete. Press enter to test: ")
    #network.test(test, testDiagnosis)
    return True, network.accuracies, network.costs
    
#from https://www.cs.cmu.edu/~112/notes/notes-variables-and-functions.html
def roundHalfUp(d):
    # Round to nearest with ties going away from zero.
    rounding = decimal.ROUND_HALF_UP
    # See other rounding options here:
    # https://docs.python.org/3/library/decimal.html#rounding-modes
    return int(decimal.Decimal(d).to_integral_value(rounding=rounding))

#The actual network class - made up of neurons and layers
class NeuralNetwork(object):
    #initialize all of the data, the number of neurons in the hidden layer,
    #the predicted outputs for each row, and the weights
    def __init__(self, trainData, trainDiagnosis, numNeurons, learningRate = 0.4):
        self.data = trainData
        self.diagnosis = trainDiagnosis["diagnosis"].tolist()
        self.costs = list()
        self.numRows = len(self.data)
        self.numNeurons = numNeurons
        self.predictedOutputs = np.zeros((self.numRows, 1))
        self.hiddenLayer = None
        self.outputLayer = None
        self.inputLayer = self.initializeInputLayerFromRow(0)
        self.hiddenLayer = self.initializeHiddenLayer()
        self.inputLayer.setNextLayer(self.hiddenLayer)
        self.outputLayer = self.initializeOutputLayer()
        self.hiddenLayer.setNextLayer(self.outputLayer)
        self.hiddenLayer.setPrevLayer(self.inputLayer)
        self.outputLayer.setPrevLayer(self.hiddenLayer)
        self.hiddenValues = list()
        self.hiddenPreSigmoidValues = list()
        self.accuracies = list()
        self.learningRate = learningRate
        self.hiddenToOutputAdjustments = list()
        #create the wegihts between the input/hidden and hidden/output
        for neuron in self.hiddenLayer.neurons:
            neuron.initializeWeights(self.inputLayer)
        for neuron in self.outputLayer.neurons:
            neuron.initializeWeights(self.hiddenLayer)

    #one run of forward propogation. Take the dot product of each neuron's weights
    #and the values of the layer before it, add the bias, then do the same
    #for the output neuron. Compare the actual value to the prediction
    def forwardPropogate(self):
        onecount = 0
        zerocount = 0
        self.outputs = list()
        self.expected = list()
        self.allHiddenAdjustments = list()
        self.allHiddenBiasAdjustments = list()
        for i in range(self.numRows):
            hiddenWeightAdjustments, hiddenBiasAdjustment = self.getValuesForInput(i)
            for j in range(self.hiddenLayer.getNumNeurons()):
                inputWeightAdjustments = self.getInputValuesForRow(i, j)
                self.hiddenLayer.neurons[j].weightAdjustments.append(inputWeightAdjustments)
            self.allHiddenAdjustments.append(hiddenWeightAdjustments)
            self.allHiddenBiasAdjustments.append(hiddenBiasAdjustment)
        accuracy = self.calcAccuracy(self.outputs, self.expected)
        print(f'train accuracy: %.4f' % accuracy)
        self.accuracies.append(accuracy)
        self.costs.append(self.calcCost(self.outputs, self.expected))
        lastCost = self.costs[-1]
        print(f"cost: %.4f" % lastCost)
        self.adjustHiddenToOutputWeights()
        self.adjustHiddenToOutputBias()
        for i in range(self.hiddenLayer.getNumNeurons()):
            self.adjustInputToHiddenWeights(i)
    
    #returns the model's accuracies
    def getAccuracies(self):
        return self.accuracies

    #each hiddenLayer neuron has its own list of desired weight updates
    #take the average of all the desired adjustments for each weight, multiply
    #by the learning rate, and subtract this value from the current weight
    #this is gradient descent
    def adjustInputToHiddenWeights(self, neuronIndex):
        adjustments = self.hiddenLayer.neurons[neuronIndex].weightAdjustments
        numWeights = len(adjustments[0])
        for i in range(numWeights):
            adjustment = 0
            for j in range(self.numRows):
                adjustment += adjustments[j][0][i]
            self.hiddenLayer.neurons[neuronIndex].weights[i] -= self.learningRate * adjustment/self.numRows

    #math for backpropagation taken from the following places:
    #https://www.cs.cmu.edu/~mgormley/courses/10601-s17/slides/lecture20-backprop.pdf
    #http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture4.pdf
    #https://datascience.stackexchange.com/questions/28719/a-good-reference-for-the-back-propagation-algorithm
    #Adjusts the input to hidden layer weights for a given neuron
    #dc/dw = dc/dneuronValue * dNeuronPostSigmoid/dNeuronPreSigmoid * dNeuronPreSigmoid/dw
    #c = cost
    #w = weight
    def inputToHiddenWeightAdjustments(self, output, expected, hiddenValue):
        adjustments = list()
        costDeriv = self.derivativeCost(output, expected)
        sigDeriv = NeuralNetwork.sigmoidDerivative(output)
        hiddenSigDeriv = NeuralNetwork.sigmoidDerivative(hiddenValue)
        for i in range(self.inputLayer.getNumNeurons()):
            inputVal = self.inputLayer.neurons[i].value
            #print(f"input val: {inputVal}")
            adjustments.append(costDeriv * sigDeriv * hiddenSigDeriv * inputVal)
        return adjustments

    #takes in a row and creates an input layer based off of the values in that row
    def getInputValuesForRow(self, row, neuronIndex):
        output = self.outputs[-1]
        expected = int(self.diagnosis[row])
        neuron = self.hiddenLayer.neurons[neuronIndex]
        adjustments = list()
        for i in range(self.inputLayer.getNumNeurons()):
            hiddenValue = self.hiddenLayer.neurons[neuronIndex].value
            adjustments.append(self.inputToHiddenWeightAdjustments(output,
            expected, hiddenValue))
        return adjustments

    #adjusts the bias from the hidden to output layer based off of
    #each training case's proposed bias adjustment
    def adjustHiddenToOutputBias(self):
        total = sum(self.allHiddenBiasAdjustments) / self.numRows
        self.outputLayer.neurons[0].bias -= total * self.learningRate
    
    #math for all backpropagation weight adjustment functions taken from the following places:
    #https://www.cs.cmu.edu/~mgormley/courses/10601-s17/slides/lecture20-backprop.pdf
    #http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture4.pdf
    #https://datascience.stackexchange.com/questions/28719/a-good-reference-for-the-back-propagation-algorithm 
    #adjust each weight from the hidden to output layer based off of 
    #each training case's proposed weight adjustment
    def adjustHiddenToOutputWeights(self):
        numWeights = len(self.allHiddenAdjustments[0])
        for i in range(numWeights):
            adjustment = 0
            for j in range(self.numRows):
                adjustment += self.allHiddenAdjustments[j][i]
            self.outputLayer.neurons[0].weights[i] -= self.learningRate * adjustment/self.numRows

    #takes in a row index and calculates the magnitudes for which each row
    #thinks the weights and bias from the hidden layer to the output
    #layer should be adjusted
    def getValuesForInput(self, row):
        self.inputLayer = self.initializeInputLayerFromRow(row)
        self.hiddenLayer.setPrevLayer(self.inputLayer)
        self.hiddenNeuronValues = list()
        self.hiddenNeuronPreSigmoid = list()
        self.outputPreSigmoid = list()
        for neuron in self.hiddenLayer.neurons:
            neuron.value = neuron.calcValue(self.inputLayer)
            self.hiddenNeuronValues.append(neuron.value)
            self.hiddenNeuronPreSigmoid.append(neuron.beforeSigmoid)
        for neuron in self.outputLayer.neurons:
            neuron.value = neuron.calcValue(self.hiddenLayer)
            output = neuron.value
            self.outputPreSigmoid.append(neuron.beforeSigmoid)
        self.expected.append(int(self.diagnosis[row]))
        self.outputs.append(output)
        weightAdjustments = self.weightAdjustmentsHiddenToOutput(output,
        int(self.diagnosis[row]), self.hiddenNeuronValues)
        biasAdjustments = self.biasAdjustmentsHiddenToOutput(output,
        int(self.diagnosis[row]))
        return weightAdjustments, biasAdjustments
    
    #derivative of the sigmoid activation function. Calculated by hand
    #and verified through WolframAlpha
    @staticmethod
    def sigmoidDerivative(value):
        return math.exp(-1 * value) / (1 + math.exp(-1 * value))**2

    #derivative of the cost function. calculated through wolfram alpha
    def derivativeCost(self, observed, expected):
        return 2 * (observed - expected)
    
    #returns the derivative of the cost function with respect to each weight
    #dc/dw = derivative of cost function * derivative of sigmoid of output function
    #* hidden value attached to that weight
    def weightAdjustmentsHiddenToOutput(self, output, expected, hiddenNeuronValues):
        adjustments = list()
        costDeriv = self.derivativeCost(output, expected)
        sigDeriv = NeuralNetwork.sigmoidDerivative(output)
        for i in range(len(hiddenNeuronValues)):
            adjustments.append(costDeriv * sigDeriv * hiddenNeuronValues[i])
        return adjustments
    
    #dc/db = derivative of cost function * derivative of sigmoid of output function
    def biasAdjustmentsHiddenToOutput(self, output, expected):
        costDeriv = self.derivativeCost(output, expected)
        sigDeriv = NeuralNetwork.sigmoidDerivative(output)
        return costDeriv*sigDeriv  

        

    #accuracy = total number of correct predictions/total number of predictions
    def calcAccuracy(self, outputs, expected):
        assert(len(outputs) == len(expected))
        correctCount = 0
        total = len(outputs)
        for i in range(total):
            if roundHalfUp(outputs[i]) == expected[i]:
                correctCount += 1
        accuracy = correctCount / total
        return accuracy
    
    #calculates the overall cost (sum of square residuals/numRows)
    def calcCost(self, outputs, expected):
        total = 0
        for i in range(len(expected)):
            total += (outputs[i] - expected[i])**2
        return total/self.numRows
    
    #trains the network, stops training if 50 epochs have been reached
    #or the network stops improving for 3 epochs
    def train(self):
        try:
            epochsWithNoImprovement = 0
            for i in range(50):
                print(f'epoch {i}')
                self.forwardPropogate()
                if len(self.costs) > 2:
                    if (self.costs[-1] - self.costs[-2]) > -0.0005:
                        epochsWithNoImprovement += 1
                if epochsWithNoImprovement == 3:
                    break
            return True, self.accuracies, self.costs
        except MemoryError:
            print("Memory error occured")
            return True, self.accuracies, self.costs

    #applies the network on external data to see the accuracy and cost on
    #this test data
    def test(self, testData, testDiagnosis):
        newHiddenLayer = copy.copy(self.hiddenLayer)
        newOutputLayer = copy.copy(self.outputLayer)
        testDiagnosis = testDiagnosis["diagnosis"].tolist()
        testOutputs = list()
        testExpected = list()
        for i in range(len(testData)):
            newInputLayer = self.initializeInputLayerFromRowForTest(testData, i, newHiddenLayer)
            for neuron in newHiddenLayer.neurons:
                neuron.value = neuron.calcValue(newInputLayer)
            for neuron in newOutputLayer.neurons:
                neuron.value = neuron.calcValue(newHiddenLayer)
                output = neuron.value
            testExpected.append(int(testDiagnosis[i]))
            testOutputs.append(output)
        accuracy = self.calcAccuracy(testOutputs, testExpected)
        print(f"test accuracy: {accuracy}")
        #scale cost to account for calcCost dividing by the number of rows in the
        #training data rather than in the test data
        #(yes, this is a bandaid fix but it works)
        cost = self.calcCost(testOutputs, testExpected) * (self.numRows/len(testData))
        print(f"test cost: {cost}")
        return (accuracy, cost)

    
    #standard sigmoid activation function - large positive values become 1
    #and large negative values become 0. Values close to 0 are in the middle
    #formula obtained from https://en.wikipedia.org/wiki/Sigmoid_function
    @staticmethod
    def sigmoidActivation(value):
        if value < -10: 
            return 0
        elif value > 10: 
            return 1
        else: 
            return 1/(1 + math.exp(-1 * value))
    
    #the initial output layer is just a layer of one neuron
    #the prev/next layer are None for now but are fixed later
    def initializeOutputLayer(self):
        neurons = [Neuron()]
        outputLayer = Layer(None, None, neurons)
        return outputLayer

    #the hidden layer contains numNeurons neurons
    def initializeHiddenLayer(self):
        neurons = list()
        for i in range(self.numNeurons):
            neurons.append(Neuron())
        hiddenLayer = Layer(self.inputLayer, self.outputLayer, neurons)
        return hiddenLayer
    
    #the input layer has no previous layer and contains one row at a time
    #of data
    def initializeInputLayerFromRow(self, rowIndex):
        row = list(self.data.iloc[rowIndex])
        neurons = list()
        for i in range(len(row)):
            neurons.append(Neuron(row[i]))
        inputLayer = Layer(None, self.hiddenLayer, neurons)
        return inputLayer
    
    #does the same thing as above but using test data instead of training data
    def initializeInputLayerFromRowForTest(self, data, rowIndex, hiddenLayer):
        row = list(data.iloc[rowIndex])
        neurons = list()
        for i in range(len(row)):
            neurons.append(Neuron(row[i]))
        inputLayer = Layer(None, hiddenLayer, neurons)
        return inputLayer


#a layer contains a list of neurons and can reference its previous and next layer
class Layer(object):
    def __init__(self, prevLayer, nextLayer, neurons):
        self.value = 0
        self.prevLayer = prevLayer
        self.nextLayer = nextLayer
        self.neurons = neurons
    
    def setPrevLayer(self, prevLayer):
        self.prevLayer = prevLayer
    
    def setNextLayer(self, nextLayer):
        self.nextLayer = nextLayer
    
    def __repr__(self):
        return f"Layer with {len(self.neurons)} neurons"

    #returns the values for every neuron in the current layer
    def getNeuronValues(self):
        values = list()
        for neuron in self.neurons:
            values.append(neuron.getValue())
        return np.array(values)
    
    #returns the number of neurons in a layer
    def getNumNeurons(self):
        return len(self.neurons)

#a neuron stores a value and has weights and a bias (offset)
class Neuron(object):
    def __init__(self, value = 0):
        self.weights = list()
        self.bias = 0.1
        self.beforeSigmoid = 0
        self.value = value
        self.weightAdjustments = list()
        self.biasAdjustments = list()

    def __repr__(self):
        return f'Neuron with value: {self.value}'    

    #once a previous layer is passed in, the neuron can randomly initialize
    #its weights based off of the number of neurons of the previous layer
    #these values are transformed so that they mostly lie between -1 and 1
    def initializeWeights(self, prevLayer):
        self.weights = np.random.randn(prevLayer.getNumNeurons())
    
    #get the sum of each weight times its corresponding value in the previous
    #layer, add the bias
    def calcValue(self, prevLayer):
        self.value = np.dot(self.weights, prevLayer.getNeuronValues())
        self.value += self.bias
        self.beforeSigmoid = self.value
        self.value = NeuralNetwork.sigmoidActivation(self.value)
        return self.value

    #changes the value of a neuron (for debugging mainly, but can also
    #be used to initialize the input layer's neurons)
    def setValue(self, value):
        self.value = value
    
    #returns the neurons's value
    def getValue(self):
        return self.value


def main():
    run()

if (__name__ == '__main__'):
    main()