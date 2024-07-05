import math
import numpy as np
from PIL import Image

# Freddie ;)
def CsvToArray():
  sigmalabels = []
  data = []
  with open("train.csv", "r") as file:
    training_data = file.reader(file)
    for row in data:
      sigmalabels.append(row[0])
      data.append(row[1:])
  return sigmalabels, data

def pngToArray():
  im = Image.open('image.png')
  na = np.array(im)
  

# JOsh
def InitialiseParams():
  pass

# TEmi
def RELU():
  pass

# TEmI
def SoftMax(weightsum):
  e_weightsum = np.exp(weightsum - np.max(weightsum))
  return e_weightsum / e_weightsum.sum()

scores = [-0.3, 0.5, 7, 1.2]
print(SoftMax(scores))

# Ollie
def SigmaWeight(fredinfoalpha, fredinfolots):
  fredtemp = 0
  fredfinal = []
  fredconstant1 = 0
  for fred in range(0, 9):
    for fred2 in range(0, 783):
      for fred3 in range(0, 783):
        fredtemp = fredtemp + fredinfoalpha[fred2]*fredinfolots[fred][fred3]
      fredconstant1 = fredconstant1 + fredtemp
      fredtemp = 0
    fredfinal.append(fredconstant1)
    fredconstant1 = 0
  return fredfinal
  
  
  
def ForwardPropagation():
  pass

def Train():
  pass
