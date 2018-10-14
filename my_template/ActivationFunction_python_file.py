import numpy as np;


class ActivationFunction:
    def sigmoid(z):
        """
        Compute the sigmoid of z

        Arguments:
        z -- A scalar or numpy array of any size.

        Return:
        s -- sigmoid(z)
        """

        s = 1 / (1 + np.exp(-z));

        return s

    def relu(z):

        r = np.maximum(z, 0);

        return r;

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right);

print(quicksort([3, 6, 8, 10, 1, 2, 1]));

#print ("sigmoid([0, 2]) = " + str(ActivationFunction.sigmoid(np.array([0,2]))))