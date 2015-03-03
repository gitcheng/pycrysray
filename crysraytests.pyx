import numpy as np
from pycrysray import *
import time
rnd = np.random
import math

# Define surfaces, wrapped crystal surface and the sensor area
surf = dict(sigdif_crys=0.1, pdif_crys=1.0, prand_crys=0.0, sigdif_wrap= 20,
            pdif_wrap=0.80, prand_wrap=0.10, 
            idx_refract_in = 1.8, idx_refract_out = 1.0, 
            sensitive=False, wrapped=True)
sensor = surf.copy()
sensor['sensitive'] = True
sensor['wrapped'] = False
sensor['idx_refract_out'] = 1.5
# Define a crystal geometry (unit=cm), which consists of a list of Plane
# objects.
rectgeom = rect_prism(center= [0,0,5], length= 10, xlen= 3.0, ylen= 3.0,
                      **surf)
hexgeom = hex_prism(center= [0,0,5], length= 10, width= 3.0, **surf)
# Define sensor geometry at two locations
sengeom1 = rectangle(center= [0, +0.7, 0], xlen= 0.9, ylen= 0.9, **sensor)
sengeom2 = rectangle(center= [0, -0.7, 0], xlen= 0.9, ylen= 0.9, **sensor)
# Create crystals. The second argument is a list of Plane objects.
cryrect = Crystal('cryrect', rectgeom + [sengeom1] + [sengeom2])
cryhex = Crystal('cryhex', hexgeom + [sengeom1] + [sengeom2])

# Create a smooth surface without wrapping
surf0 = dict(sigdif_crys=0.0, pdif_crys=1.0, prand_crys=0.0,
            idx_refract_in = 2.0, idx_refract_out = 1.0, 
             sensitive=False, wrapped=False)
# Create a cubic crystal with surf0
cubegeom = rect_prism(center= [0,0,0], length= 2, xlen= 2, ylen= 2,
                      **surf0)
crycube = Crystal('crycube', cubegeom)


def check_plane():
    '''
    Perform some checks on a Plane object
    '''
    corners = [[0,0,0],[0,1,0.5],[1,1,0.5],[1,0,0]]
    print '(1): Print out surface property. The plane has corners:'
    print corners
    print 'and surface property'
    print surf
    print 'The direction of the normal should be [0, -0.447.., 0.894...]'
    print 
    tpl = Plane(corners, **surf)
    tpl.print_properties()

    print '\n(2): Check if a point is on the plane (on_plane())'
    pt1 = np.array([0.3, 0.4, 0.20])
    print '  Point', pt1, tpl.on_plane(pt1)
    pt2 = np.array([0.3, 0.4, 0.21])
    print '  Point', pt2, tpl.on_plane(pt2)
    pt3 = np.array([0.6, 1.1, 0.55])
    print '  Point', pt3, tpl.on_plane(pt3)

    print '\n(3): Check if a point is on the plane and in bound (contain())'
    print '  Point', pt1, tpl.contain(pt1)
    print '  Point', pt2, tpl.contain(pt2)
    print '  Point', pt3, tpl.contain(pt3)

    print '\n(4): Check if a point is on the same side as normal'+\
        '(normal_side())'
    pt22 = np.array([0.3, 0.4, 0.19])
    print '  Point', pt2, tpl.normal_side(pt2)
    print '  Point', pt22, tpl.normal_side(pt22)

    print '\n(5): Some speed tests'
    count = 10000
    start = time.clock()
    for i in xrange(count): tpl.on_plane(pt3)
    end = time.clock()
    print 'on_plane():  %.3f usec' % ((end-start)/count *1e6)
    start = time.clock()
    for i in xrange(count): tpl.contain(pt3)
    end = time.clock()
    print 'contain():  %.3f usec' % ((end-start)/count *1e6)
    start = time.clock()
    for i in xrange(count): tpl.normal_side(pt3)
    end = time.clock()
    print 'normal_side():  %.3f usec' % ((end-start)/count *1e6)

    print '\n(6): Random points on the surface, must all be contained'
    p = np.zeros(3, dtype=DTYPE)
    ok = True
    count = 10000
    start = time.clock()
    for i in xrange(count):
        p[0] = rnd.uniform() * 1.0
        p[1] = rnd.uniform() * 1.0
        p[2] = 0.5 * p[1]
        if not tpl.contain(p):
            ok = False
    end = time.clock()
    print 'All %d points are ok:' %(count), ok
    print 'time:  %.3f msec' % ((end-start) *1e3)


def check_crystal():
    '''
    Perform some checks for a Crystal object.
    '''
    print '(1) Generate random points that should all be inside a'+\
        'rectanglar crystal'
    count = 10000
    n = 0
    p = np.zeros(3, dtype=DTYPE)
    start = time.clock()
    for i in xrange(count):
        p[0] = rnd.uniform(-1.5, 1.5)
        p[1] = rnd.uniform(-1.5, 1.5)
        p[2] = rnd.uniform(0, 10)
        if cryrect.contain(p):
            n += 1
    end = time.clock()
    print 'Points inside %d/%d= %.2f%%' % (n, count, n/float(count)*100)
    print 'time:  %.3f msec' % ((end-start) *1e3)

    print '\n(2) Generate random points that part of them are inside'+\
        'a hexagonal crystal'
    count = 10000
    n = 0
    p = np.zeros(3, dtype=DTYPE)
    start = time.clock()
    for i in xrange(count):
        p[0] = rnd.uniform(-2, 2)
        p[1] = rnd.uniform(-2, 2)
        p[2] = rnd.uniform(-1, 11)
        if cryhex.contain(p):
            n += 1
    end = time.clock()
    print 'Points inside %d/%d= %.2f%%' % (n, count, n/float(count)*100)
    print 'time:  %.3f msec' % ((end-start) *1e3)
    rv = (math.sqrt(3)/2 * 3*3 * 10) / (4*4*12) * 100
    print 'The fraction should be compared to ',
    print '(sqrt(3)/2 * 3^2*10) / (4*4*12) = %.2f%%'%(rv)

def check_photon():
    '''
    Test some photon properties
    '''
    print 'Surface of the crystal for test (1) and (2)'
    crycube.planes[0].print_properties()

    print '(1), small incident angle'
    x = np.array([0, 0, 0], dtype=DTYPE)
    d = np.array([1, 0.2, 0.3], dtype=DTYPE)
    photon1 = Photon(x, d, t=0, trackvtx= True, mfp = 2.0)
    print 'Before propagation'
    photon1.print_properties()
    photon1.propagate(crycube)
    print 'After propagation'
    photon1.print_properties()

    print '(2), larger incident angle'
    x = np.array([0.5, -0.5, 0], dtype=DTYPE)
    d = np.array([1, 1, 0.0], dtype=DTYPE)
    photon2 = Photon(x, d, t=0, trackvtx= True, mfp = 2.0)
    print 'Before propagation'
    photon2.print_properties()
    photon2.propagate(crycube)
    print 'After propagation'
    photon2.print_properties()
    print 'vertices:', photon2.vertices

    print '\n(3) timing test using a hexagonal crystal with two sensors'
    count = 1000
    ndet = 0
    tsum = 0
    start = time.clock()
    for i in range(count):
        x, d = generate_p6(np.array([0,0,9], dtype=DTYPE), 0.1, 0.1)
        photon3 = Photon(x, d, t=0, trackvtx= False, mfp = 50)
        photon3.propagate(cryhex)
        if photon3.status == photon3.transmitted and \
           photon3.lastplane.sensitive :
            ndet += 1
            tsum += photon3.t - photon3.t0
    print 'photon collection %d/%d = %.2f%%' % (ndet,count,ndet/float(count)*100)
    print ' average time to reach a sensor = %.2f ns' % (tsum/ndet)
    end = time.clock()
    print 'cpu time:  %.3f msec per photon' % ((end-start)/count *1e3)


def main():
    import sys
    if not len(sys.argv) == 2:
        raise ValueError('Exactly one argument is needed')

    rnd.seed(0)
    arg = sys.argv[1]
    if arg == 'check_plane':
        check_plane()
    elif arg == 'check_crystal':
        check_crystal()
    elif arg == 'check_photon':
        check_photon()

if __name__ == '__main__':
    main()
