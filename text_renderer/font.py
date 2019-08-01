import os
import random
import math
import glob

import cv2
from PIL import Image
import numpy as np
import scipy.cluster
from scipy import ndimage


this_dir, _ = os.path.split(__file__)


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

    def __init__(self, font_dir=f'{this_dir}/data/font', font_list=f'{this_dir}/data/font/fontlist.txt'):
        self.fonts = [os.path.join(font_dir, f.strip()) for f in open(font_list)]

    def get_sample(self):
        """
        Samples from the font state distribution
        """
        return {
            'font': self.fonts[int(np.random.randint(0, len(self.fonts)))],
            'size': self.size[1]*np.random.randn() + self.size[0],
            # 'underline': np.random.rand() < self.underline,
            'underline': False,
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
    def __init__(self, imfn=random.choice(glob.glob(f'{this_dir}/data/fill'))):
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


class AffineTransformState(object):
    """
    Defines the random state for an affine transformation
    """
    proj_type = Image.AFFINE
    rotation = [0, 5]  # rotate normal dist mean, std
    skew = [0, 0]  # skew normal dist mean, std

    def sample_transformation(self, imsz):
        theta = math.radians(self.rotation[1]*np.random.randn() + self.rotation[0])
        ca = math.cos(theta)
        sa = math.sin(theta)
        R = np.zeros((3,3))
        R[0,0] = ca
        R[0,1] = -sa
        R[1,0] = sa
        R[1,1] = ca
        R[2,2] = 1
        S = np.eye(3,3)
        S[0,1] = math.tan(math.radians(self.skew[1]*np.random.randn() + self.skew[0]))
        A = R @ S
        x = imsz[1]/2
        y = imsz[0]/2
        return (A[0,0], A[0,1], -x*A[0,0] - y*A[0,1] + x,
                A[1,0], A[1,1], -x*A[1,0] - y*A[1,1] + y)


class PerspectiveTransformState(object):
    """
    Defines teh random state for a perspective transformation
    Might need to use http://stackoverflow.com/questions/14177744/how-does-perspective-transformation-work-in-pil
    """
    proj_type = Image.PERSPECTIVE
    a_dist = [1, 0.01]
    b_dist = [0, 0.005]
    c_dist = [0, 0.005]
    d_dist = [1, 0.01]
    e_dist = [0, 0.0005]
    f_dist = [0, 0.0005]

    def v(self, dist):
        return dist[1]*np.random.randn() + dist[0]

    def sample_transformation(self, imsz):
        x = imsz[1]/2
        y = imsz[0]/2
        a = self.v(self.a_dist)
        b = self.v(self.b_dist)
        c = self.v(self.c_dist)
        d = self.v(self.d_dist)
        e = self.v(self.e_dist)
        f = self.v(self.f_dist)

        # scale a and d so scale kept same
        #a = 1 - e*x
        #d = 1 - f*y

        z = -e*x - f*y + 1
        A = np.zeros((3,3))
        A[0,0] = a + e*x
        A[0,1] = b+f*x
        A[0,2] = -a*x-b*y-e*x*x-f*x*y+x
        A[1,0] = c+e*y
        A[1,1] = d+f*y
        A[1,2] = -c*x-d*y-e*x*y-f*y*y+y
        A[2,0] = e
        A[2,1] = f
        A[2,2] = z
        # print a,b,c,d,e,f
        # print z
        A = A / z

        return (A[0,0], A[0,1], A[0,2], A[1,0], A[1,1], A[1,2], A[2,0], A[2,1])


class ElasticDistortionState(object):
    """
    Defines a random state for elastic distortions
    """
    displacement_range = 1
    alpha_dist = [[15, 30], [0, 2]]
    sigma = [[8, 2], [0.2, 0.2]]
    min_sigma = [4, 0]

    def sample_transformation(self, imsz):
        choices = len(self.alpha_dist)
        c = int(np.random.randint(0, choices))
        sigma = max(self.min_sigma[c], np.abs(self.sigma[c][1]*np.random.randn() + self.sigma[c][0]))
        alpha = np.random.uniform(self.alpha_dist[c][0], self.alpha_dist[c][1])
        dispmapx = np.random.uniform(-1*self.displacement_range, self.displacement_range, size=imsz)
        dispmapy = np.random.uniform(-1*self.displacement_range, self.displacement_range, size=imsz)
        dispmapx = alpha * ndimage.gaussian_filter(dispmapx, sigma)
        dispmaxy = alpha * ndimage.gaussian_filter(dispmapy, sigma)
        return dispmapx, dispmaxy


class DistortionState(object):
    blur = [0, 1]
    sharpen = 0
    sharpen_amount = [30, 10]
    noise = 4
    resample = 0.1
    resample_range = [24, 32]

    def get_sample(self):
        return {
            'blur': np.abs(self.blur[1]*np.random.randn() + self.blur[0]),
            'sharpen': np.random.rand() < self.sharpen,
            'sharpen_amount': self.sharpen_amount[1]*np.random.randn() + self.sharpen_amount[0],
            'noise': self.noise,
            'resample': np.random.rand() < self.resample,
            'resample_height': int(np.random.uniform(self.resample_range[0], self.resample_range[1]))
        }

class SurfaceDistortionState(DistortionState):
    noise = 8
    resample = 0
