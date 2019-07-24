import os
import random

import cv2
import numpy as np
import scipy.cluster


class FontState(object):
    """
    Defines the random state of the font rendering
    """
    size = [60, 10]  # normal dist mean, std
    underline = 0.05
    strong = 0.5
    oblique = 0.2
    wide = 0.5
    strength = [0.02778, 0.05333]  # uniform dist in this interval
    underline_adjustment = [1.0, 2.0]  # normal dist mean, std
    kerning = [2, 5, 0, 20]  # beta distribution alpha, beta, offset, range (mean is a/(a+b))
    border = 0.25
    random_caps = 1.0
    capsmode = [str.lower, str.upper, str.capitalize]  # lower case, upper case, proper noun
    curved = 0.2
    random_kerning = 0.2
    random_kerning_amount = 0.1

    def __init__(self, font_dir='./data/font', font_list='./data/font/fontlist.txt'):
        self.fonts = [os.path.join(font_dir, f.strip()) for f in open(font_list)]

    def get_sample(self):
        """
        Samples from the font state distribution
        """
        return {
            'font': self.fonts[int(np.random.randint(0, len(self.fonts)))],
            'size': self.size[1]*np.random.randn() + self.size[0],
            'underline': np.random.rand() < self.underline,
            'underline_adjustment': max(2.0, min(-2.0, self.underline_adjustment[1] * np.random.randn()
                                                 + self.underline_adjustment[0])),
            'strong': np.random.rand() < self.strong,
            'oblique': np.random.rand() < self.oblique,
            'strength': (self.strength[1] - self.strength[0])*np.random.rand() + self.strength[0],
            'char_spacing': int(self.kerning[3]*(np.random.beta(self.kerning[0], self.kerning[1]))
                                + self.kerning[2]),
            'border': np.random.rand() < self.border,
            'random_caps': np.random.rand() < self.random_caps,
            'capsmode': random.choice(self.capsmode),
            'curved': np.random.rand() < self.curved,
            'random_kerning': np.random.rand() < self.random_kerning,
            'random_kerning_amount': self.random_kerning_amount,
        }


class ColorState(object):
    """
    Gives the foreground, background, and optionally border colourstate.
    Does this by sampling from a training set of images,
    and clustering in to desired number of colours
    (http://stackoverflow.com/questions/3241929/python-find-dominant-most-common-color-in-an-image)
    """
    def __init__(self, imfn='./data/ali.jpg'):
        self.im = cv2.imread(imfn, 0)

    def get_sample(self, n_colours):
        a = self.im.flatten().astype(np.float32)
        codes, dist = scipy.cluster.vq.kmeans(a, n_colours)
        # get std of centres
        vecs, dist = scipy.cluster.vq.vq(a, codes)
        colours = []
        for i in range(n_colours):
            try:
                code = codes[i]
                std = np.std(a[vecs == i])
                colours.append(std * np.random.randn() + code)
            except IndexError:
                print("\tcolor error")
                colours.append(int(sum(colours)/float(len(colours))))
        # choose randomly one of each colour
        return np.random.permutation(colours)


class BaselineState(object):
    curve = lambda this, a: lambda x: a*x*x
    differential = lambda this, a: lambda x: 2*a*x
    a = [0, 0.1]

    def get_sample(self):
        """
        Returns the functions for the curve and differential for a and b
        """
        a = self.a[1] * np.random.randn() + self.a[0]
        return {
            'curve': self.curve(a),
            'diff': self.differential(a),
        }


class BorderState(object):
    outset = 0.5
    width = [4, 4]  # normal dist
    position = [[0,0], [-1,-1], [-1,1], [1,1], [1,-1]]

    def get_sample(self):
        p = self.position[int(np.random.randint(0,len(self.position)))]
        w = max(1, int(self.width[1]*np.random.randn() + self.width[0]))
        return {
            'outset': np.random.rand() < self.outset,
            'width': w,
            'position': [int(-1*np.random.uniform(0,w*p[0]/1.5)), int(-1*np.random.uniform(0,w*p[1]/1.5))]
        }
