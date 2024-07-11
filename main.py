import math
import numpy as np
from numpy import asarray
from PIL import Image
import random,csv

# Freddie ;)
"""
Converts a CSV of image pixels into an array of pixel values
INPUTS: CSV File name
RETURNS: 2d array of all the pixel values for each image [[pixels in img 1], [pixels in img 2], ..]
"""
def CsvToArray(filename="train.csv"):
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

"""
Converts a PNG (must be 28x28, greyscale) file to an array of pixel values
INPUTS: PNG File
RETURNS: 2d array of all the pixel values in the img [pixel0, pixel1, .., pixel783]
"""
# Might be good eventually to have the option of either entering either a single png file, or a folder of pngs? lmk what you think
def pngToArray(file):
  img = Image.open(file)
  numpydata = asarray(img)
  numpydata.tolist()
  pixels_array = []
  for row in numpydata:
    for pixel in row:
      pixels_array.append(sum(pixel)//3)
  return pixels_array

# JOsh
def InitialiseParams():
  pass

# TEmi
def RELU(x):
  return x if x > 0 else 0

# TEmI
def SoftMax(weightsum):
  e_weightsum = np.exp(weightsum - np.max(weightsum))
  return e_weightsum / e_weightsum.sum()

#scores = [-0.3, 0.5, 7, 1.2]
#print(SoftMax(scores))

# Ollie
"""
Performs the weighted sum operation for a whole layer, simeltaneously
INPUTS: Input array, 2d array of weights for each ouput node [[weights for node 0], [weights for node 1], ..]
RETURNS: array of the weighted sum for each node in next layer [sum for node 0, sum for node 2, ..]
"""
def SigmaWeight(fredinputsalpha, fredweightsalpha):
  fredtemp = 0
  fredfinal = [] # Sums at each node (size = number of nodes in next layer)
  fredconstant1 = 0
  # For each node in the layer:
  for fred in range(10):
    # For each input in the previous layer:
    for fred2 in range(784):
      # Multiply an input by all its weights, and add them together
      fredtemp = fredtemp + fredinputsalpha[fred2]*fredweightsalpha[fred][fred2] # Weighted sum at node N = I1*w1 + I2*w2 + ... + In*wn
    fredconstant1 = fredconstant1 + fredtemp
    fredtemp = 0
    # Add the total sum at that node to array
    fredfinal.append(fredconstant1)
    fredconstant1 = 0
  return fredfinal

"""
Creates a n*m list of random nums betweeen 0 and 1, to be used as  weights or a whole layer.
Used on first forward prop, to initialise weights ready for training
INPUTS: int number of nodes in previous layer (tells us how many weights there should be going into each node), n
        int number of nodes in current layer (tells us how many sets of weights we're going to need), m
RETURNS: n*m list of random nums 0 <-> 1
"""
def getRandomWeights(noPreviousLayerNodes: int, noCurrentLayerNodes: int) -> list:
  return [[random.random() for i in range(noPreviousLayerNodes)] for i in range(noCurrentLayerNodes)] # What's the advantage of using a numpy array here? I feel like there should be one

"""
Performs singular forward pass through network
INPUTS: list input pixels, list hidden layer weights, list output layer weights
RETURNS: int predicted number
Also stores all values at each step in the network, to be passed into backpropogation
""" 
def ForwardPropagation(input: list, hiddenWeights=getRandomWeights(784, 10), outputWeights=getRandomWeights(10, 784)):
  
  hiddenWeightedSums = SigmaWeight(input, hiddenWeights)
  hiddenActivations = map(RELU, hiddenWeightedSums)
  
  samples = [(random.randint(0, 10), random.randint(0, 783)) for i in range(3)]
  for sample in samples:
    print(f"Input : {input[sample[0]][sample[1]]}")
    print(f"Weight : {hiddenWeights[sample[0]][sample[1]]}")
    print(f"Weighted Sum : {hiddenWeightedSums[sample[0]]}")
    print(f"Activation : {hiddenActivations[sample]}")
  """
  ouputWeightedSum
  ouputActivation

  prediction
  """

def Train():
  pass


#def main():
  #ForwardPropagation()

#main()
#pngToArray("colours.png")
label, data = CsvToArray("mnist_train.csv")
print(label)
