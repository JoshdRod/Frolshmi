from layersController import getRandomWeights
import numpy as np
from numpy import asarray

# Defines a Layer (Use HiddenLayer or OutputLayer for backprop functionality)
class BaseLayer:
  # Data
  backPropConstant = 1.5 # Placeholder (used in weight update formula)

  inputSize = 0 # no. nodes in previous layer
  layerSize = 0 # no. nodes in layer

  inputs = []
  weights = []
  weightedSums = []
  activations = []
  pdErrorWRTActivation = []
  pdActivationWRTWeightedSum = []
  pdErrorWRTWeightedSum = []
  pdWeightedSumWRTWeights = []
  pdErrorWRTWeights = []
  

  # Functionality

  # Josh
  def __init__(self, inputSize, layerSize, backPropConstant=0.5):
    self.backPropConstant = backPropConstant

    self.inputSize = inputSize
    self.layerSize = layerSize

    self.weights = getRandomWeights(self.inputSize, self.layerSize) # [node 1 weights[], node 2 weights[], ..]


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
  Functionality defined in Child classes
  INPUTS: type
  RETURNS: Call to correct function based on type
  """
  def __Activation__(self):
      return -1 # Occurs if Layer is instantiated, and not Hidden/OutputLayer

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

  """
  Updates weights of layer using update rule
  current rule (FIX) : w_n = w_o + backpropConstant*(dError / dw_o), where backpropConstant starts at 1.5, and decreases by 0.02 every cycle until it = 0. 
  INPUTS: Layer
  OUPUTS: New weights for layer
  """
  def UpdateWeights(self):
    for i in range(self.layerSize):
      for j in range(self.inputSize):
        self.weights[i][j] += self.backPropConstant*(self.pdErrorWRTWeights[i][j])
    self.backPropConstant -= 0.02 if self.backPropConstant > 0 else 0

class HiddenLayer(BaseLayer):
  def __Activation__(self):
      return self.__RELU__()
  
  """
  Finds error gradient of each neuron in hidden layer
  INPUTS: Error of neurons in proceeding layer, weights of proceeding layer
  OUTPUTS: error gradient of each neuron in layer
  """
  def BackPropogate(self, proceedingLayer: BaseLayer):

    # 1 - Find dError / da_k
    self.pdErrorWRTActivation = [sum([proceedingLayer.weights[i][j] * proceedingLayer.pdErrorWRTWeightedSum[i] for i in range(proceedingLayer.layerSize)]) for j in range(self.layerSize)]
    print(self.pdErrorWRTActivation)

    # 2 - Find da_k / dz_k
    self.pdActivationWRTWeightedSum = [(activation)*(1 - activation) for activation in self.activations]
    print(self.pdActivationWRTWeightedSum)

    # 3 - Find dError / dz_k
    self.pdErrorWRTWeightedSum = [self.pdErrorWRTActivation[i] * self.pdActivationWRTWeightedSum[i] for i in range(self.layerSize)]
    print(self.pdErrorWRTWeightedSum)

    # 4 - Find dz_k / dw_k,l
    # Forgot how to do this : setting it to sum of all weighted sums / weights # TODO FIX!!!! (This is also going to break if any weight is == 0, but should be okay for now)
    self.pdWeightedSumWRTWeights = [[sum(self.weightedSums) / self.weights[i][j] for j in range(self.inputSize)] for i in range(self.layerSize)]

    print(f"Weighted summation WRT Weights in hidden layer : ")
    for e in self.pdWeightedSumWRTWeights:
      print(e, end="\n\n\n\n")
    print("\n\n\n\n")

    # 5 - Find dError / dw_k,l
    self.pdErrorWRTWeights = [[self.pdWeightedSumWRTWeights[i][j] * self.pdErrorWRTWeightedSum[i] for j in range(self.inputSize)] for i in range(self.layerSize)]
    
    print(f"Error WRT Weights in hidden layer : ")
    for e in self.pdErrorWRTWeights:
      print(e, end="\n\n\n\n")

  
class OutputLayer(BaseLayer):
  def __Activation__(self):
      return self.__SoftMax__()
  
  def BackPropogate(self, expectedOutput):
    # Convert expected value to an array of 'probablities'
    # e.g '4' -> [0,0,0,0,1,0,0,0,0,0]
    expectedOutputProbs = [0 for i in range(expectedOutput)] + [1]+ [0 for i in range(self.layerSize - expectedOutput - 1)]
    print(expectedOutputProbs)

    # 1 - Calculate dError / da_k
    self.pdErrorWRTActivation = [expectedOutputProbs[i] - self.activations[i] for i in range(self.layerSize)] 
    print(self.pdErrorWRTActivation)

    # 2 - Calculate da_k / dz_k
    self.pdActivationWRTWeightedSum = [(activation)*(1 - activation) for activation in self.activations]
    print(self.pdActivationWRTWeightedSum)

    # 3 - Calculate dError / dz_k
    self.pdErrorWRTWeightedSum = [self.pdErrorWRTActivation[i] * self.pdActivationWRTWeightedSum[i] for i in range(self.layerSize)]
    print(f"Partial rate of change of Error WRT the weighted sum at each output node : {self.pdErrorWRTWeightedSum}")

     # 4 - Find dz_k / dw_k,l
    # Forgot how to do this : setting it to sum of all weighted sums / weights # TODO FIX!!!! (This is also going to break if any weight is == 0, but should be okay for now)
    self.pdWeightedSumWRTWeights = [[sum(self.weightedSums) / self.weights[i][j] for j in range(self.inputSize)] for i in range(self.layerSize)]

    print(f"Weighted summation WRT Weights in hidden layer : ")
    for e in self.pdWeightedSumWRTWeights:
      print(e, end="\n\n\n\n")
    print("\n\n\n\n")
    
    # 5 - Find dError / dw_k,l
    self.pdErrorWRTWeights = [[self.pdWeightedSumWRTWeights[i][j] * self.pdErrorWRTWeightedSum[i] for j in range(self.inputSize)] for i in range(self.layerSize)]
    
    print(f"Error WRT Weights in hidden layer : ")
    for e in self.pdErrorWRTWeights:
      print(e, end="\n\n\n\n")

