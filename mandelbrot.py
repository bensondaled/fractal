##
# convert mand.gif -coalesce -duplicate 1,-2-1 -quiet -layers OptimizePlus -loop 0 mandloop.gif

from multiprocessing import Process
import numpy as np
import matplotlib.pyplot as pl
from soup.animation import Animation

def mandelbrot(x, y, threshold):
    """Calculates whether the number c = x + i*y belongs to the 
    Mandelbrot set. In order to belong, the sequence z[i + 1] = z[i]**2 + c
    must not diverge after 'threshold' number of steps. The sequence diverges if the absolute value of z[i+1] is greater than 4.
    
    :param float x: the x component of the initial complex number
    :param float y: the y component of the initial complex number
    :param int threshold: the number of iterations to considered it converged
    """

    # initial conditions
    c = complex(x, y)
    z = complex(0, 0)

    for i in range(threshold):
        z = z**2 + c
        if abs(z) > 4.0:  # it diverged
            return i
        
    return threshold - 1  # it didn't diverge

def draw_img(center, wh, thresh, npix=200, arr=None):
    re0, im0 = center
    w = h = wh

    im = np.linspace(im0-h/2, im0+h/2, npix)
    re = np.linspace(re0-w/2, re0+w/2, npix)

    if arr is None:
        arr = np.empty((len(im), len(re)))

    for i in range(len(im)):
        for j in range(len(re)):
            R, I = re[j], im[i]
            arr[i, j] = mandelbrot(R, I, thresh)
    return arr, (im, re)

#center=(-1.2963247639177446, 0.44182080996569695)
#center = (-0.46475777464800705, -0.5450620084753436)

thresh = 800
npix = 800
zoom_init = 2
center = (-1.2500579833374386, 0.00549374971639973)
im, coords = draw_img(center=center,
                      wh=zoom_init, #w/h in complex plane coords
                      thresh=thresh,
                      npix=npix,
                      arr=None)
y0, y1, x0, x1 = coords[0][0], coords[0][-1], coords[1][0], coords[1][-1]

fig = pl.figure(figsize=(6,6))
ax = fig.add_axes([0, 0, 1, 1])
img = ax.imshow(im[::-1], cmap=pl.cm.inferno_r,
                extent=(x0, x1, y0, y1))

##
zooms = np.logspace(0.3, -15, 150)
an = Animation('mandel', fig=fig, rate=5.0)
for zoom in zooms:
    print(zoom)
    im, coords = draw_img(center=center,
                          wh=zoom,
                          arr=im,
                          thresh=thresh,
                          npix=npix)
    img.set_data(im[::-1])
    img.set_clim(im.min(), im.max())
    an.put()
#an.end()

proc = Process(target=print_func, args=(name,))
procs.append(proc)
                proc.start()

                    # complete the processes
                        for proc in procs:
                                    proc.join()


##
