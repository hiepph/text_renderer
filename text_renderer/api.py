import math
import os
import random
import uuid
import glob

import pygame
from pygame import freetype
import numpy as np
from scipy import ndimage
import cv2
from PIL import Image
from tqdm import tqdm

from .font import FontState, ColorState, BaselineState, BorderState, AffineTransformState, PerspectiveTransformState, SurfaceDistortionState, DistortionState

this_dir, _ = os.path.split(__file__)


MJBLEND_NORMAL = "normal"
MJBLEND_ADD = "add"
MJBLEND_SUB = "subtract"
MJBLEND_MULT = "multiply"
MJBLEND_MULTINV = "multiplyinv"
MJBLEND_SCREEN = "screen"
MJBLEND_DIVIDE = "divide"
MJBLEND_MIN = "min"
MJBLEND_MAX = "max"

pygame.init()


fontstate = FontState()
baselinestate = BaselineState()
affinestate = AffineTransformState()
perspectivestate = PerspectiveTransformState()
diststate = DistortionState()
surfdiststate = SurfaceDistortionState()


def global_distortions(arr):
    # http://scipy-lectures.github.io/advanced/image_processing/#image-filtering
    ds = diststate.get_sample()

    blur = ds['blur']
    sharpen = ds['sharpen']
    sharpen_amount = ds['sharpen_amount']
    noise = ds['noise']

    newarr = np.minimum(np.maximum(0, arr + np.random.normal(0, noise, arr.shape)), 255)
    if blur > 0.1:
        newarr = ndimage.gaussian_filter(newarr, blur)
    if sharpen:
        newarr_ = ndimage.gaussian_filter(arr, blur/2)
        newarr = newarr + sharpen_amount*(newarr - newarr_)

    if ds['resample']:
        sh = newarr.shape[0]
        newarr = resize_image(newarr, newh=ds['resample_height'])
        newarr = resize_image(newarr, newh=sh)

    return newarr


def surface_distortions(arr):
    ds = surfdiststate.get_sample()
    blur = ds['blur']

    origarr = arr.copy()
    arr = np.minimum(np.maximum(0, arr + np.random.normal(0, ds['noise'], arr.shape)), 255)
    # make some changes to the alpha
    arr[...,1] = ndimage.gaussian_filter(arr[...,1], ds['blur'])
    ds = surfdiststate.get_sample()
    arr[...,0] = ndimage.gaussian_filter(arr[...,0], ds['blur'])
    if ds['sharpen']:
        newarr_ = ndimage.gaussian_filter(origarr[...,0], blur/2)
        arr[...,0] = arr[...,0] + ds['sharpen_amount']*(arr[...,0] - newarr_)

    return arr


class FillImageState(object):
    """
    Handles the images used for filling the background, foreground, and border surfaces
    """
    blend_amount = [0.0, 0.25]  # normal dist mean, std
    blend_modes = [MJBLEND_NORMAL, MJBLEND_ADD, MJBLEND_MULTINV, MJBLEND_SCREEN, MJBLEND_MAX]
    blend_order = 0.5
    min_textheight = 16.0  # minimum pixel height that you would find text in an image

    def __init__(self, data_dir=f'{this_dir}/data/fill'):
        self.data_dir = data_dir
        self.im_list = os.listdir(data_dir)

    def get_sample(self, surfarr):
        """
        The image sample returned should not have it's aspect ratio changed, as this would never happen in real world.
        It can still be resized of course.
        """
        # load image
        imfn = os.path.join(self.data_dir, random.choice(self.im_list))
        baseim = np.array(Image.open(imfn))

        # choose a colour channel or rgb2gray
        if baseim.ndim == 3:
            if np.random.rand() < 0.25:
                baseim = rgb2gray(baseim)
            else:
                baseim = baseim[..., np.random.randint(0,3)]
        else:
            assert(baseim.ndim == 2)

        imsz = baseim.shape
        surfsz = surfarr.shape

        # don't resize bigger than if at the original size, the text was less than min_textheight
        max_factor = float(surfsz[0])/self.min_textheight
        # don't resize smaller than it is smaller than a dimension of the surface
        min_factor = max(float(surfsz[0] + 5)/float(imsz[0]), float(surfsz[1] + 5)/float(imsz[1]))
        # sample a resize factor
        factor = max(min_factor, min(max_factor, ((max_factor-min_factor)/1.5)*np.random.randn() + max_factor))
        sampleim = resize_image(baseim, factor)
        imsz = sampleim.shape
        # sample an image patch
        good = False
        curs = 0
        while not good:
            curs += 1
            if curs > 1000:
                print("difficulty getting sample")
                break
            try:
                x = np.random.randint(0,imsz[1]-surfsz[1])
                y = np.random.randint(0,imsz[0]-surfsz[0])
                good = True
            except ValueError:
                # resample factor
                factor = max(min_factor, min(max_factor, ((max_factor-min_factor)/1.5)*np.random.randn() + max_factor))
                sampleim = resize_image(baseim, factor)
                imsz = sampleim.shape
        imsample = (np.zeros(surfsz) + 255).astype(surfarr.dtype)
        imsample[...,0] = sampleim[y:y+surfsz[0],x:x+surfsz[1]]
        imsample[...,1] = surfarr[...,1].copy()

        return {
            'image': imsample,
            'blend_mode': random.choice(self.blend_modes),
            'blend_amount': min(1.0, np.abs(self.blend_amount[1]*np.random.randn() + self.blend_amount[0])),
            'blend_order': np.random.rand() < self.blend_order,
        }


def rgb2gray(rgb):
    # RGB -> grey-scale (as in Matlab's rgb2grey)
    try:
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    except IndexError:
        try:
            gray = rgb[:,:,0]
        except IndexError:
            gray = rgb[:,:]
    return gray


def resize_image(im, r=None, newh=None, neww=None, filtering=Image.BILINEAR):
    dt = im.dtype
    I = Image.fromarray(im)
    if r is not None:
        h = im.shape[0]
        w = im.shape[1]
        newh = int(round(r*h))
        neww = int(round(r*w))
    if neww is None:
        neww = int(newh*im.shape[1]/float(im.shape[0]))
    if newh > im.shape[0]:
        I = I.resize([neww, newh], Image.ANTIALIAS)
    else:
        I.thumbnail([neww, newh], filtering)
    return np.array(I).astype(dt)


def add_fillimage(arr, fillimstate=FillImageState()):
    """
    Adds a fill image to the array.
    For blending this might be useful:
    - http://stackoverflow.com/questions/601776/what-do-the-blend-modes-in-pygame-mean
    - http://stackoverflow.com/questions/5605174/python-pil-function-to-divide-blend-two-images
    """
    fis = fillimstate.get_sample(arr)

    image = fis['image']
    blend_mode = fis['blend_mode']
    blend_amount = fis['blend_amount']
    blend_order = fis['blend_order']

    # change alpha of the image
    if blend_amount > 0:
        if blend_order:
            image = image.astype(np.float64)
            image[...,1] *= blend_amount
            arr = grey_blit(image, arr, blend_mode=blend_mode)
        else:
            arr = arr.astype(np.float64)
            arr[...,1] *= (1 - blend_amount)
            arr = grey_blit(arr, image, blend_mode=blend_mode)

    return arr


def grey_blit(src, dst, blend_mode=MJBLEND_NORMAL):
    """
    This is for grey + alpha images
    """
    # http://stackoverflow.com/a/3375291/190597
    # http://stackoverflow.com/a/9166671/190597
    # blending with alpha http://stackoverflow.com/questions/1613600/direct3d-rendering-2d-images-with-multiply-blending-mode-and-alpha
    # blending modes from: http://www.linuxtopia.org/online_books/graphics_tools/gimp_advanced_guide/gimp_guide_node55.html
    dt = dst.dtype
    src = src.astype(np.single)
    dst = dst.astype(np.single)
    out = np.empty(src.shape, dtype = 'float')
    alpha = np.index_exp[:, :, 1]
    rgb = np.index_exp[:, :, 0]
    src_a = src[alpha]/255.0
    dst_a = dst[alpha]/255.0
    out[alpha] = src_a+dst_a*(1-src_a)
    old_setting = np.seterr(invalid = 'ignore')
    src_pre = src[rgb]*src_a
    dst_pre = dst[rgb]*dst_a
    # blend:
    blendfuncs = {
        MJBLEND_NORMAL: lambda s, d, sa_: s + d*sa_,
        MJBLEND_ADD: lambda s, d, sa_: np.minimum(255, s + d),
        MJBLEND_SUB: lambda s, d, sa_: np.maximum(0, s - d),
        MJBLEND_MULT: lambda s, d, sa_: s*d*sa_ / 255.0,
        MJBLEND_MULTINV: lambda s, d, sa_: (255.0 - s)*d*sa_ / 255.0,
        MJBLEND_SCREEN: lambda s, d, sa_: 255 - (1.0/255.0)*(255.0 - s)*(255.0 - d*sa_),
        MJBLEND_DIVIDE: lambda s, d, sa_: np.minimum(255, d*sa_*256.0 / (s + 1.0)),
        MJBLEND_MIN: lambda s, d, sa_: np.minimum(d*sa_, s),
        MJBLEND_MAX: lambda s, d, sa_: np.maximum(d*sa_, s),
    }
    out[rgb] = blendfuncs[blend_mode](src_pre, dst_pre, (1-src_a))
    out[rgb] /= out[alpha]
    np.seterr(**old_setting)
    out[alpha] *= 255
    np.clip(out,0,255)
    # astype('uint8') maps np.nan (and np.inf) to 0
    out = out.astype(dt)
    return out


def imcrop(arr, rect):
    if arr.ndim > 2:
        return arr[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2],...]
    else:
        return arr[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]


def get_bb(arr, eq=None):
    if eq is None:
        v = np.nonzero(arr > 0)
    else:
        v = np.nonzero(arr == eq)
    xmin = v[1].min()
    xmax = v[1].max()
    ymin = v[0].min()
    ymax = v[0].max()
    return [xmin, ymin, xmax-xmin, ymax-ymin]


def arr_scroll(arr, dx, dy):
    arr = np.roll(arr, dx, axis=1)
    arr = np.roll(arr, dy, axis=0)
    return arr


def get_bordershadow(bg_arr, colour, borderstate=BorderState()):
    """
    Gets a border/shadow with the movement state [top, right, bottom, left].
    Inset or outset is random.
    """
    bs = borderstate.get_sample()
    outset = bs['outset']
    width = bs['width']
    position = bs['position']

    # make a copy
    border_arr = bg_arr.copy()
    # re-colour
    border_arr[...,0] = colour
    if outset:
        # dilate black (erode white)
        border_arr[...,1] = ndimage.grey_dilation(border_arr[...,1], size=(width, width))
        border_arr = arr_scroll(border_arr, position[0], position[1])

        # canvas = 255*n.ones(bg_arr.shape)
        # canvas = grey_blit(border_arr, canvas)
        # canvas = grey_blit(bg_arr, canvas)
        # pyplot.imshow(canvas[...,0], cmap=cm.Greys_r)
        # pyplot.show()

        return border_arr, bg_arr
    else:
        # erode black (dilate white)
        border_arr[...,1] = ndimage.grey_erosion(border_arr[...,1], size=(width, width))
        return bg_arr, border_arr


def get_ga_image(surf):
    r = pygame.surfarray.pixels_red(surf)
    a = pygame.surfarray.pixels_alpha(surf)
    r = r.reshape((r.shape[0], r.shape[1], 1))
    a = a.reshape(r.shape)
    return np.concatenate((r, a), axis=2).swapaxes(0, 1)


def apply_perspective_arr(arr, affstate, a_proj_type, perstate, p_proj_type, filtering=Image.BICUBIC):
    img = Image.fromarray(arr)
    img = img.transform(img.size, a_proj_type,
                        affstate,
                        filtering)
    img = img.transform(img.size, p_proj_type,
                        perstate,
                        filtering)
    arr = np.array(img)
    return arr


def gen(text, sz=(800, 200),
        color=random.choice(glob.glob(f'{this_dir}/data/fill/*')),
        fill=f'{this_dir}/data/fill',
        substring_crop=0, random_crop=True):
    """Generate text image from input text
    """
    fs = fontstate.get_sample()

    # clear background
    bg_surf = pygame.Surface((round(2.0 * fs['size'] * len(text)),
                              sz[1]), pygame.SRCALPHA, 32)

    font = freetype.Font(fs['font'], int(fs['size']))
    # random params
    text = fs['capsmode'](text) if fs['random_caps'] else text
    font.underline = fs['underline']
    font.underline_adjustment = fs['underline_adjustment']
    font.strong = fs['strong']
    font.oblique = fs['oblique']
    font.strength = fs['strength']
    char_spacing = fs['char_spacing']

    font.antialiased = True
    font.origin = True

    colorstate = ColorState(color)
    cs = colorstate.get_sample(2 + fs['border'])

    mid_idx = int(math.floor(len(text) / 2))
    curve = [0 for c in text]
    rotations = [0 for c in text]
    if fs['curved'] and len(text) > 1:
        bs = baselinestate.get_sample()
        for i, c in enumerate(text[mid_idx+1:]):
            curve[mid_idx+i+1] = bs['curve'](i+1)
            rotations[mid_idx+i+1] = -int(math.degrees(math.atan(bs['diff'](i+1)/float(fs['size']/2))))
        for i, c in enumerate(reversed(text[:mid_idx])):
            curve[mid_idx-i-1] = bs['curve'](-i-1)
            rotations[mid_idx-i-1] = -int(math.degrees(math.atan(bs['diff'](-i-1)/float(fs['size']/2))))
        mean_curve = sum(curve) / float(len(curve)-1)
        curve[mid_idx] = -1 * mean_curve

    # render text (centered)
    char_bbs = []
    # place middle char
    rect = font.get_rect(text[mid_idx])
    rect.centerx = bg_surf.get_rect().centerx
    rect.centery = bg_surf.get_rect().centery + rect.height
    rect.centery += curve[mid_idx]
    bbrect = font.render_to(bg_surf, rect, text[mid_idx], rotation=rotations[mid_idx])

    bbrect.x = rect.x
    bbrect.y = rect.y - rect.height
    char_bbs.append(bbrect)

    # render chars to the right
    last_rect = rect
    for i, c in enumerate(text[mid_idx+1:]):
        char_fact = 1.0
        if fs['random_kerning']:
            char_fact += fs['random_kerning_amount'] * np.random.randn()
        newrect = font.get_rect(c)
        newrect.y = last_rect.y
        newrect.topleft = (last_rect.topright[0] + char_spacing*char_fact, newrect.topleft[1])
        newrect.centery = max(0 + newrect.height*1, min(sz[1] - newrect.height*1, newrect.centery + curve[mid_idx+i+1]))
        try:
            bbrect = font.render_to(bg_surf, newrect, c, rotation=rotations[mid_idx+i+1])
        except ValueError:
            bbrect = font.render_to(bg_surf, newrect, c)
        bbrect.x = newrect.x
        bbrect.y = newrect.y - newrect.height
        char_bbs.append(bbrect)
        last_rect = newrect

    # render chars to the left
    last_rect = rect
    for i, c in enumerate(reversed(text[:mid_idx])):
        char_fact = 1.0
        if fs['random_kerning']:
            char_fact += fs['random_kerning_amount']*np.random.randn()
        newrect = font.get_rect(c)
        newrect.y = last_rect.y
        newrect.topright = (last_rect.topleft[0] - char_spacing*char_fact, newrect.topleft[1])
        newrect.centery = max(0 + newrect.height*1, min(sz[1] - newrect.height*1, newrect.centery + curve[mid_idx-i-1]))
        try:
            bbrect = font.render_to(bg_surf, newrect, c, rotation=rotations[mid_idx-i-1])
        except ValueError:
            bbrect = font.render_to(bg_surf, newrect, c)
        bbrect.x = newrect.x
        bbrect.y = newrect.y - newrect.height
        char_bbs.append(bbrect)
        last_rect = newrect


    bg_arr = get_ga_image(bg_surf)

    # colour text
    bg_arr[..., 0] = cs[0]

    # border/shadow
    if fs['border']:
        l1_arr, l2_arr = get_bordershadow(bg_arr, cs[2])
    else:
        l1_arr = bg_arr

     # do rotation and perspective distortion
    affstate = affinestate.sample_transformation(l1_arr.shape)
    perstate = perspectivestate.sample_transformation(l1_arr.shape)
    l1_arr[...,1] = apply_perspective_arr(l1_arr[...,1], affstate, affinestate.proj_type, perstate, perspectivestate.proj_type)
    if fs['border']:
        l2_arr[..., 1] = apply_perspective_arr(l2_arr[...,1], affstate,affinestate.proj_type, perstate, perspectivestate.proj_type)

    # get bb of text
    if fs['border']:
        bb = pygame.Rect(get_bb(grey_blit(l2_arr, l1_arr)[...,1]))
    else:
        bb = pygame.Rect(get_bb(l1_arr[...,1]))

    if random_crop:
        bb.inflate_ip(10*np.random.randn()+15, 10*np.random.randn()+15)
    else:
        inflate_amount = int(0.4*bb[3])
        bb.inflate_ip(inflate_amount, inflate_amount)

    # crop image
    l1_arr = imcrop(l1_arr, bb)
    if fs['border']:
        l2_arr = imcrop(l2_arr, bb)

    canvas = (255*np.ones(l1_arr.shape)).astype(l1_arr.dtype)
    canvas[..., 0] = cs[1]

    # add in natural images
    canvas = add_fillimage(canvas, FillImageState(fill))
    l1_arr = add_fillimage(l1_arr)
    if fs['border']:
        l2_arr = add_fillimage(l2_arr)

    # add per-surface distortions
    l1_arr = surface_distortions(l1_arr)
    if fs['border']:
        l2_arr = surface_distortions(l2_arr)

    # compose global image
    blend_modes = [MJBLEND_NORMAL, MJBLEND_ADD, MJBLEND_MULTINV, MJBLEND_SCREEN, MJBLEND_MAX]
    count = 0
    while True:
        globalcanvas = grey_blit(l1_arr, canvas, blend_mode=random.choice(blend_modes))
        if fs['border']:
            globalcanvas = grey_blit(l2_arr, globalcanvas, blend_mode=random.choice(blend_modes))
        globalcanvas = globalcanvas[...,0]
        std = np.std(globalcanvas.flatten())
        count += 1
        if std > 20:
            break
        if count > 10:
            print("\tERR: can't get good contrast")
            return None, None
    canvas = globalcanvas

    # add global distortions
    canvas = global_distortions(canvas)

    # noise removal
    canvas = ndimage.filters.median_filter(canvas, size=(3,3))

    return canvas, text
