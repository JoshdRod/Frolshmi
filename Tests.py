import random
class Tester:
    def TestWeightedSums(previousLayer, currentLayerWeights, currentLayerWeightedSums):
        # Test samples of the network
        sample = random.randint(0, 783)

        print(f"Input: {previousLayer}\n\n\n")
        print(f"Weights: {currentLayerWeights[sample]}\n\n\n")
        print(f"Weighted sum: {currentLayerWeightedSums[sample]}")