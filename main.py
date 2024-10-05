# TODO:
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
  return data

class Network:
  # Data
  layers : list = []
  output : int = None

  # Functionality
  def __init__(self):
    self.layers : list = [HiddenLayer(inputSize=784, layerSize=10),
                   OutputLayer(inputSize=10, layerSize=10)]

  """
  TODO : Add comment to method
  """
  def ForwardPropogate(self, expectedInput : list):
    previousLayerInput = expectedInput
    for layer in self.layers:
      previousLayerInput = layer.ForwardPropogate(previousLayerInput)
    
    # Find index of largest probability in output (that's the guessed number)
    self.output = previousLayerInput.index(max(previousLayerInput))
      
  """
  Backpropogates through layers
  Each layer calculates the partial derivative of the Error between the generated output and the expected output WRT each weight,
  then updates the weight using the defined rule
  INPUTS: list backpropogrand = value to backpropogate to previous layer
  """
  def BackPropogate(self, backPropogrand : list):
    self.layers.reverse()
    for layer in self.layers: # So, now output layer is at front
      layer.BackPropogate(backPropogrand)
      layer.UpdateWeights()
      backPropogrand = layer
    self.layers.reverse() # Flip them back before F-Prop (!!BODGE!!)

def __main__():
  # ==============================================
  # Get network running on 100 different images
  # - Output its guess
  # - Output the actual value
  # - Backpropogate error through network
  # - At end, show the difference in accuracy between the first 20 entries, and the last 20 entries
  # ==============================================
  totalSuccessRate = 0
  first20SuccessRate = 0
  last20SuccessRate = 0

  # Initialise Network
  N = Network()

  # Get images
  Images = CsvToArray()
  # Iterate over images
  for i in range(100):
    # Propogate over image
    N.ForwardPropogate(Images[i][1:])

    # Output guess + actual
    print(f"Generated Ouput : {N.output}")
    print(f"Expected Output : {Images[i][0]}")

    if N.output == Images[i][0]:
      print("Successful guess!")
      # Update success rate
      totalSuccessRate += 1
      if i < 20: first20SuccessRate += 1
      elif i >= 79: last20SuccessRate += 1 # As loop goes 0 - 99

    # Backpropogate
    N.BackPropogate(Images[i][0])
  # Output total success rate, 1st 20 success rate, last 20 success rate
  print(f"Final Stats : \nTotal Success Rate : {totalSuccessRate}\nFirst 20 Success Rate : {first20SuccessRate}\nLast 20 Success Rate : {last20SuccessRate}")


if __name__ == "__main__":
  __main__()