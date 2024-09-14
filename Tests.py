import random
class Tester:
    def TestLayer(self, previousLayer, currentLayerWeights, currentLayerWeightedSums, currentLayerActivation):
        # Test samples of the network
        currentLayerSize = len(currentLayerWeights)
        sample = random.randint(0, currentLayerSize - 1) # As randint is INCLUSIVE on both sides

        print(f"Input: {previousLayer}\n\n\n")
        print(f"Weights on node {sample}: {currentLayerWeights[sample]}\n\n\n")
        print(f"Weighted sum on node {sample}: {currentLayerWeightedSums[sample]}")
        print(f"Activation of node {sample}: {currentLayerActivation[sample]}\n\n\n")