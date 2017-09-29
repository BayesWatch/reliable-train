
import numpy.random as npr

print("some other stuff")

if __name__ == '__main__':
    # return random scores and occasionally throw errors
    if npr.rand() < 0.95:
        print(npr.rand())
    else:
        raise ValueError("Dummy error, you better catch it!")

