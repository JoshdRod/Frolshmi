# TODO:
# Softmax doesn't give array where values add up to 1. Fix this.
# Implement backpropogation, and updating weights
# Allow network to forward prop on multiple images, one after another
# Allow network to parse + test on user-made images (both from file, and from camera)
#

import math

from PIL import Image
import random
import Tests
import csv

from layers import HiddenLayer, OutputLayer


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
    self.layers.reverse() 
    for layer in self.layers: # So, now output layer is at front
      layer.BackPropogate(self.input[1][0][0])

def __main__():
  N = Network()
  N.ForwardPropogate()
  N.BackPropogate()


if __name__ == "__main__":
  __main__()