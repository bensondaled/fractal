##
# convert *.png mand.gif
# convert mand.gif -coalesce -duplicate 1,-2-1 -quiet -layers OptimizePlus -loop 0 mandloop.gif

from multiprocessing import Process
import logging
import numpy as np
import matplotlib.pyplot as pl
pl.ioff()
import os

logging.basicConfig(filename='log.log', level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logging.getLogger('matplotlib.font_manager').disabled = True

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

def generate_img(center, wh, thresh, npix=200, arr=None):
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

# Params
thresh = 800
npix = 800
zooms = np.logspace(0.3, -15, 150)
zooms = np.array([zooms, np.arange(len(zooms))]).T
#center=(-1.2963247639177446, 0.44182080996569695)
#center = (-0.46475777464800705, -0.5450620084753436)
center = (-1.2500579833374386, 0.00549374971639973)
save_dir = f'imgs/{center[0]}-{center[1]}_{zooms[0,0]}-{zooms[-1,0]}_{thresh}_{npix}'
os.makedirs(save_dir)
n_procs = 8

def display_images(zooms, center, thresh, npix, cmap=pl.cm.magma_r):
    # create a baseline image object
    im, coords = generate_img(center=center,
                          wh=zooms[0,0], #w/h in complex plane coords
                          thresh=thresh,
                          npix=npix,
                          arr=None)
    y0, y1, x0, x1 = coords[0][0], coords[0][-1], coords[1][0], coords[1][-1]

    fig = pl.figure(figsize=(6,6))
    ax = fig.add_axes([0, 0, 1, 1])
    img = ax.imshow(im[::-1], cmap=cmap,
                    extent=(x0, x1, y0, y1))

    for zoom, zoom_id in zooms:
        zoom_id = int(zoom_id)
        logging.info(f'Running {zoom_id} = {zoom} (batch {zooms[0,1]}-{zooms[-1,1]})')
        im, coords = generate_img(center=center,
                              wh=zoom,
                              arr=im,
                              thresh=thresh,
                              npix=npix)
        img.set_data(im[::-1])
        img.set_clim(im.min(), im.max())
        fig.savefig(os.path.join(save_dir, f'{zoom_id:05}.png'))

zlists = np.array_split(zooms, n_procs, axis=0)
procs = []
for i, zlist in enumerate(zlists):
    proc = Process(target=display_images, args=(zlist, center, thresh, npix))
    procs.append(proc)
    proc.start()

for proc in procs:
    proc.join()

##
