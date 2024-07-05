import math
import numpy as np
from PIL import Image

# Freddie ;)
"""
Converts a CSV of image pixels into an array of pixel values
INPUTS: CSV File name
RETURNS: 2d array of all the pixel values for each image [[pixels in img 1], [pixels in img 2], ..]
"""
def CsvToArray():
  sigmalabels = []
  data = []
  with open("train.csv", "r") as file: # Might be good to make the file name a param, and default it to the "train.csv" (I belive there's a programming principle which involves this idea, but I can't remember it!)
    training_data = file.reader(file)
    for row in data:
      sigmalabels.append(row[0])
      data.append(row[1:])
  return sigmalabels, data

"""
Converts a PNG (must be 28x28, greyscale) file to an array of pixel values
INPUTS: PNG File
RETURNS: 2d array of all the pixel values in the img [pixel0, pixel1, .., pixel783]
"""
# Might be good eventually to have the option of either entering either a single png file, or a folder of pngs? lmk what you think
def pngToArray():
  im = Image.open('image.png')
  na = np.array(im)
  

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
   
def ForwardPropagation():
  pass

def Train():
  pass


def main():
  x = SigmaWeight([3]*784, [[2]*784]*10)
  print(len(x))
  print(x)

main()