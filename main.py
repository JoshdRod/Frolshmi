# TODO:
# Softmax doesn't give array where values add up to 1. Fix this.
# Implement backpropogation, and updating weights
# Allow network to forward prop on multiple images, one after another
# Allow network to parse + test on user-made images (both from file, and from camera)
#

import math
import numpy as np
from numpy import asarray
from PIL import Image
import random
import Tests
import csv

"""
Creates a n*m list of random nums betweeen 0 and 1, to be used as  weights or a whole layer.
Used on first forward prop, to initialise weights ready for training
INPUTS: int number of nodes in previous layer (tells us how many weights there should be going into each node), n
        int number of nodes in current layer (tells us how many sets of weights we're going to need), m
RETURNS: n*m list of random nums 0 <-> 1
"""
def getRandomWeights(noPreviousLayerNodes: int, noCurrentLayerNodes: int) -> list:
  return [[random.uniform(-1, 1) for i in range(noPreviousLayerNodes)] for i in range(noCurrentLayerNodes)] # random.uniform selects a random real number between a and b inclusive

# Freddie ;)
"""
Converts a CSV of image pixels into an array of pixel values
INPUTS: CSV File name
RETURNS: 2d array of all the pixel values for each image [[pixels in img 1], [pixels in img 2], ..]
"""
def CsvToArray(filename="mnist_train.csv"):
  data = []
  with open(filename, "r") as file: # Might be good to make the file name a param, and default it to the "train.csv" (I belive there's a programming principle which involves this idea, but I can't remember it!)
    training_data = csv.reader(file)
    counter = 0 
    for row in training_data:
      if counter == 0:
        sigmalabels = row
      else:
        int_row = [int(i) for i in row]
        data.append(int_row)
      counter += 1
  return sigmalabels, data

class Network:
  # Data
  layers = []
  input = []

  # Functionality
  def __init__(self):
    self.layers = [Layer(inputSize=784, layerSize=10),
                   Layer(inputSize=10, layerSize=10, type="output")]
    self.input = CsvToArray()

  def ForwardPropogate(self):
    previousLayerInput = self.input[1][1] # TEMP - Picks 2nd number in array 
    for layer in self.layers:
      previousLayerInput = layer.ForwardPropogate(previousLayerInput)
    
    print(previousLayerInput)
      

  def BackPropogate():
    pass

class Layer:
  # Data
  type = "" # Hidden or output (changes the activation func used)
  backPropConstant = 0 # Placeholder (used in weight update formula)

  inputSize = 0 # no. nodes in previous layer
  layerSize = 0 # no. nodes in layer

  inputs = []
  weights = []
  weightedSums = []
  activations = []
  pdErrorWRTActivation = []
  pdActivationWRTWeightedSum = []
  pdErrorWRTWeightedSum = []

  # Functionality

  # Josh
  def __init__(self, inputSize, layerSize, type="hidden", backPropConstant=0.5):
    self.type = type
    self.backPropConstant = backPropConstant

    self.inputSize = inputSize
    self.layerSize = layerSize

    self.weights = getRandomWeights(self.inputSize, self.layerSize)


  # Ollie
  """
  Performs the weighted sum operation for a whole layer, simeltaneously
  INPUTS: Input array, 2d array of weights for each ouput node [[weights for node 0], [weights for node 1], ..]
  RETURNS: array of the weighted sum for each node in next layer [sum for node 0, sum for node 2, ..]
  """
  def __WeightedSummation__(self):
    fredtemp = 0
    fredfinal = [] # Sums at each node (size = number of nodes in next layer)
    fredconstant1 = 0
    # For each node in the layer:
    for fred in range(self.layerSize):
      # For each input in the previous layer:
      for fred2 in range(self.inputSize):
        # Multiply an input by all its weights, and add them together
        fredtemp = fredtemp + self.inputs[fred2]*self.weights[fred][fred2] # Weighted sum at node N = I1*w1 + I2*w2 + ... + In*wn
      fredconstant1 = fredconstant1 + fredtemp
      fredtemp = 0
      # Add the total sum at that node to array
      fredfinal.append(fredconstant1)
      fredconstant1 = 0
    return fredfinal

  # Temi
  """
  Performs RELU on all weighted sums in the layer
  INPUTS: weightedSums
  RETURNS: List of RELU(weightedSum) for each weightedSum in weightedSums
  """
  def __RELU__(self):
    return list(map(lambda x: x if x > 0 else 0, self.weightedSums))
    
  # Temi
  """
  Performs a softmax operation over each element in a list
  See full explaination of softmax here: https://victorzhou.com/blog/softmax/
  The general concept is, we can find the probability of a value being chosen, even if it's negative, because e^n > 0, even for -ive nums
  INPUTS: weightedSums
  RETURNS: list, of same size, of probabilities of each number being the one in the image 
  """
  def __SoftMax__(self):
    e_weightedSum = np.exp(self.weightedSums - np.max(self.weightedSums))
    return list(e_weightedSum / e_weightedSum.sum()) 

  # Josh
  """
  Selects the correct activation function to be performed based on the layer's type
  If layer is a hidden layer, uses RELU, if layer is the output layer, uses SoftMax
  INPUTS: type
  RETURNS: Call to correct function based on type
  """
  def __Activation__(self):
    if self.type == "hidden":
      return self.__RELU__()
    if self.type == "output":
      return self.__SoftMax__()
    
    else: # Only occurs if layer type is something other than hidden or output - which should never happen
      return -1

  # Josh
  """
  Begins forward propogation
  INPUTS: Input activations
  Correctly applies a weighted summation to inputs, then runs weighted summations through an activation function
  """
  def ForwardPropogate(self, inputs: list[int]):
    # Assign inputs to layer, for use in weighted summation + backprop
    self.inputs = inputs

    # Weighted Summation
    self.weightedSums = self.__WeightedSummation__()
    # Activation
    self.activations = self.__Activation__()
    return self.activations

  def BackPropogate():
    pass

  def UpdateWeights():
    pass


def __main__():
  N = Network()
  print(N.ForwardPropogate())

if __name__ == "__main__":
  __main__()


# # Freddie ;)
# """
# Converts a CSV of image pixels into an array of pixel values
# INPUTS: CSV File name
# RETURNS: 2d array of all the pixel values for each image [[pixels in img 1], [pixels in img 2], ..]
# """
# def CsvToArray(filename="mnist_train.csv"):
#   data = []
#   with open(filename, "r") as file: # Might be good to make the file name a param, and default it to the "train.csv" (I belive there's a programming principle which involves this idea, but I can't remember it!)
#     training_data = csv.reader(file)
#     counter = 0 
#     for row in training_data:
#       if counter == 0:
#         sigmalabels = row
#       else:
#         int_row = [int(i) for i in row]
#         data.append(int_row)
#       counter += 1
#   return sigmalabels, data

# """
# Converts a PNG (must be 28x28, greyscale) file to an array of pixel values
# INPUTS: PNG File
# RETURNS: 2d array of all the pixel values in the img [pixel0, pixel1, .., pixel783]
# """
# # Might be good eventually to have the option of either entering either a single png file, or a folder of pngs? lmk what you think
# def pngToArray(file):
#   img = Image.open(file)
#   numpydata = asarray(img)
#   numpydata.tolist()
#   pixels_array = []
#   for row in numpydata:
#     for pixel in row:
#       pixels_array.append(sum(pixel)//3)
#   return pixels_array


# # TEmi
# def RELU(x):
#   return x if x > 0 else 0

# # TEmI
# """
# Performs a softmax operation over each element in a list
# See full explaination of softmax here: https://victorzhou.com/blog/softmax/
# The general concept is, we can find the probability of a value being chosen, even if it's negative, because e^n > 0, even for -ive nums
# INPUTS: list of raw weighted sums for each number
# RETURNS: list, of same size, of probabilities of each number being the one in the image 
# """
# def SoftMax(weightsum):
#   e_weightsum = np.exp(weightsum - np.max(weightsum))
#   return list(e_weightsum / e_weightsum.sum()) 

# #scores = [-0.3, 0.5, 7, 1.2]
# #print(SoftMax(scores))

# # Ollie
# """
# Performs the weighted sum operation for a whole layer, simeltaneously
# INPUTS: Input array, 2d array of weights for each ouput node [[weights for node 0], [weights for node 1], ..]
# RETURNS: array of the weighted sum for each node in next layer [sum for node 0, sum for node 2, ..]
# """
# def SigmaWeight(fredinputsalpha, fredweightsalpha):
#   fredtemp = 0
#   fredfinal = [] # Sums at each node (size = number of nodes in next layer)
#   fredconstant1 = 0
#   # For each node in the layer:
#   for fred in range(10):
#     # For each input in the previous layer:
#     for fred2 in range(784):
#       # Multiply an input by all its weights, and add them together
#       fredtemp = fredtemp + fredinputsalpha[fred2]*fredweightsalpha[fred][fred2] # Weighted sum at node N = I1*w1 + I2*w2 + ... + In*wn
#     fredconstant1 = fredconstant1 + fredtemp
#     fredtemp = 0
#     # Add the total sum at that node to array
#     fredfinal.append(fredconstant1)
#     fredconstant1 = 0
#   return fredfinal

# """
# Creates a n*m list of random nums betweeen 0 and 1, to be used as  weights or a whole layer.
# Used on first forward prop, to initialise weights ready for training
# INPUTS: int number of nodes in previous layer (tells us how many weights there should be going into each node), n
#         int number of nodes in current layer (tells us how many sets of weights we're going to need), m
# RETURNS: n*m list of random nums 0 <-> 1
# """
# def getRandomWeights(noPreviousLayerNodes: int, noCurrentLayerNodes: int) -> list:
#   return [[random.uniform(-1, 1) for i in range(noPreviousLayerNodes)] for i in range(noCurrentLayerNodes)] # random.uniform selects a random real number between a and b inclusive
# # Josh
# """
# Performs singular forward pass through network
# INPUTS: list input pixels, list hidden layer weights, list output layer weights
# RETURNS: int predicted number
# Also stores all values at each step in the network, to be passed into backpropogation
# """ 
# def ForwardPropagation(input: list, Tester=Tests.Tester(), hiddenWeights=getRandomWeights(784, 10), outputWeights=getRandomWeights(10, 10)):
  
#   hiddenWeightedSums = SigmaWeight(input, hiddenWeights)
#   hiddenActivations = list(map(RELU, hiddenWeightedSums))

#   Tester.TestLayer(input, hiddenWeights, hiddenWeightedSums, hiddenActivations)

  
#   # STILL NEED TO TEST THESE
#   outputWeightedSums = SigmaWeight(input, hiddenWeights)
#   outputActivations = SoftMax(outputWeightedSums)

#   Tester.TestLayer(hiddenActivations, outputWeights, outputWeightedSums, outputActivations)
  
#   prediction = outputActivations.index(max(outputActivations))
  
#   print(outputActivations)
#   print(prediction)
  
# # JOsh
# def InitialiseParams():
#   pass

# def Train():
#   pass

# def main():
#   T = Tests.Tester()
#   inputs = CsvToArray()
#   # For now, just create a random list of 784 numbers
#   #randomInputs = [random.random() for i in range(784)]
#   ForwardPropagation(inputs[1][1])
#   print("Actual:", inputs[0][1])

# main()
# #pngToArray("colours.png")