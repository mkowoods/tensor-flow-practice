import pylab as plt
import json
import numpy as np

IMG_SHAPE = (28, 28)
#misses = json.load(open('misses.json', 'rb'))

class Misses:
    
    def __init__(self, misses_path):
        self.misses = json.load(open(misses_path, 'rb'))
        self.data = self.misses['data']
        self.labels = self.misses['label']
        self.guesses = self.misses['guess']
        
    @property
    def count(self):
        return len(self.data)
    
    def get_img(self, idx):
        
        
        img = np.reshape(self.data[idx], IMG_SHAPE)
        label = self.labels[idx]
        guess = self.guesses[idx]
        
        return (label, guess, img)
    


def get_img_by_idx(idx):
    return np.reshape(misses['data'][idx], IMG_SHAPE)
    

def plot_image(img):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(img)
    plt.show()



misses = Misses('misses.json')
    
    