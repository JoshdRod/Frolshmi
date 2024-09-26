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

from layers import HiddenLayer, OutputLayer

"""
Creates a n*m list of random nums betweeen 0 and 1, to be used as  weights for a whole layer.
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
    self.layers = [HiddenLayer(inputSize=784, layerSize=10),
                   OutputLayer(inputSize=10, layerSize=10)]
    self.input = CsvToArray()

  def ForwardPropogate(self):
    previousLayerInput = self.input[1][1] # TEMP - Picks 2nd number in array 
    for layer in self.layers:
      previousLayerInput = layer.ForwardPropogate(previousLayerInput)
    
    print(previousLayerInput)
      

  def BackPropogate(self):
    pass

def __main__():
  N = Network()
  print(N.ForwardPropogate())

if __name__ == "__main__":
  __main__()