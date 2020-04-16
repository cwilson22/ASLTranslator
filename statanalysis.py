import matplotlib.pyplot as plt
import numpy as np

def plot(hist):
    vhist = [h.cpu().numpy() for h in hist]
    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1,len(vhist)),vhist,label="Validation")
    plt.ylin((0,1.))
    plt.xticks(np.arange(1, len(vhist)+1,1.0))
    plt.legend()
    plt.show()


# vhist = [0.9568,0.9695,0.9686,0.9736,0.9770,0.9819,0.9751,0.9828,0.9767,0.9814,0.9815,0.9824,0.9841,0.9846,0.9826]
# thist = [0.7905,0.8955,0.9073,0.9144,0.9244,0.9250,0.9281,0.9308,0.9333,0.9357,0.9347,0.9367,0.9390,0.9394,0.9388]
#
# plt.title("Validation Accuracy vs. Number of Training Epochs")
# plt.xlabel("Training Epochs")
# plt.ylabel("Validation Accuracy")
# plt.plot(range(1,len(vhist)+1),vhist,label="Validation")
# plt.plot(range(1,len(thist)+1),thist,label="Training")
# plt.ylim((0,1.))
# plt.xticks(np.arange(1, len(thist)+1,1.0))
# plt.legend()
# plt.show()