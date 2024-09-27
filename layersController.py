import random
"""
Creates a n*m list of random nums betweeen 0 and 1, to be used as  weights for a whole layer.
Used on first forward prop, to initialise weights ready for training
INPUTS: int number of nodes in previous layer (tells us how many weights there should be going into each node), n
        int number of nodes in current layer (tells us how many sets of weights we're going to need), m
RETURNS: n*m list of random nums 0 <-> 1
"""
def getRandomWeights(noPreviousLayerNodes: int, noCurrentLayerNodes: int) -> list:
  return [[random.uniform(-1, 1) for i in range(noPreviousLayerNodes)] for i in range(noCurrentLayerNodes)] # random.uniform selects a random real number between a and b inclusive
