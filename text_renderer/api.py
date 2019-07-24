import math
import random

import pygame
from pygame import freetype
import numpy as np
from scipy import ndimage
import cv2

from .font import FontState, ColorState, BaselineState, BorderState


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


# def imcrop(arr, rect):
#     if arr.ndim > 2:
#         return arr[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2],...]
#     else:
#         return arr[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]


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


def gen(text, sz=(800, 200),
        fontstate=FontState(), colorstate=ColorState(), baselinestate=BaselineState()):
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
    rect.centery +=  curve[mid_idx]
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
    # if fs['border']:
        # l1_arr, l2_arr = get_bordershadow(bg_arr, cs[2])
    # else:
        # l1_arr = bg_arr


    l1_arr, l2_arr = get_bordershadow(bg_arr, cs[2])

    canvas = (255*np.ones(l1_arr.shape)).astype(l1_arr.dtype)
    canvas[...,0] = cs[1]

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
            print("\tcan't get good contrast")
            return None
    canvas = globalcanvas
    cv2.imwrite('test.jpg', canvas)
