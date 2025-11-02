import random
import numpy as np
def main():
    # TODO : SELECT THE ALGORITHM 
    # TODO : COMPUTE THE TIME OF ALGO 
    # TODO : OUTPUT SHOULD BE SEND TO GUI
    goal = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]]

    print(GenerateInput())








def GenerateInput():
    numbers = list(range(9))
    random.shuffle(numbers)
    matrix = np.array(numbers).reshape(3, 3)
    return matrix

if __name__ == "__main__":
    main()
