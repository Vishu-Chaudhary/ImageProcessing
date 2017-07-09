import numpy as np


class SearchEngine:
    def __init__(self, index):
        # store index of searched images
        self.index = index

    def engine(self, queryImageFeatures):
        results = {}

        # looping over index file

        for (filename, features) in self.index.items():
            # compute the chi-squared distance between the features
            # in our index and our query features -- using the
            # chi-squared distance which is normally used in the
            # computer vision field to compare histograms
            distance = self.chi2_distance(features, queryImageFeatures)
            # now that we have the distance between the two feature
            # vectors, we can udpate the results dictionary -- the
            # key is the current image ID in the index and the
            # value is the distance we just computed, representing
            # how 'similar' the image in the index is to our query
            results[filename] = distance

        # now soting to take relavent images are at first
        results = sorted([(features, filename) for (filename, features) in results.items()])
        return results

    @staticmethod
    def chi2_distance(hist1, hist2, eps=1e-10):
        distance = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(hist1, hist2)])

        return distance
