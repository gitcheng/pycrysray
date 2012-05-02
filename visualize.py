#!/usr/bin/env python
'''
Draw a prism and internal photon path
'''
import numpy as np
import sys
from pycrysray import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
arr = np.array


def main():

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('--geo', dest='geo', default='rect',\
                          help='Geometry option: "rect", "trap"')
    parser.add_option('--seed', dest='seed', default=31416, help='Random seed')
    parser.add_option('--refidx', dest='refidx', default='',\
                          help='Reflectivity properties: three parameters are needed, random_reflect_probability:diffuse_reflect_probability:diffuse_sigma(degree)')
    parser.add_option('--mfp', dest='mfp', default=1000, help='Mean free path')
    parser.add_option('--verbose', dest='verbose', action='store_true', default=False)
    (opts, args) = parser.parse_args()
    
    ## Random number seed
    np.random.seed(int(opts.seed))

    # Surface properties
    refpars = opts.refidx.split(':')
    ref= {'random_reflect': float(refpars[0]),
          'diffuse_reflect':float(refpars[1]),
          'diffuse_sigma':float(refpars[2])} 

    plist = None
    if ( opts.geo == 'rect' ):
        center= arr([0,0,0])
        dims = arr([1.5, 1.5, 5.5])  # half lengths
        plist = rect_prism(center, dims, **ref)

    elif ( opts.geo == 'trap' ):
        # Define crystal planes with corners
        cn1 = arr([-1.6, -1, -6])
        cn2 = arr([+1.6, -1, -6])
        cn3 = arr([-1.2, +1, -6])
        cn4 = arr([+1.2, +1, -6])
        cn5 = arr([-2.0, -1.25, +6])
        cn6 = arr([+2.0, -1.25, +6])
        cn7 = arr([-1.5, +1.25, +6])
        cn8 = arr([+1.5, +1.25, +6])
        cs1 = arr([ cn1, cn2, cn4, cn3] )
        cs2 = arr([ cn5, cn6, cn8, cn7] )
        cs3 = arr([ cn1, cn2, cn6, cn5] )
        cs4 = arr([ cn4, cn3, cn7, cn8] )
        cs5 = arr([ cn2, cn4, cn8, cn6] )
        cs6 = arr([ cn1, cn5, cn7, cn3] )

        plist = [Plane(cs1,**ref),Plane(cs2,**ref),Plane(cs3,**ref),Plane(cs4,**ref),Plane(cs5,**ref),Plane(cs6,**ref)]

    else:
        sys.exit('Unknown geometry '+opts.geo)

    # Create photons
    rloc = np.random.random( (10, 3) ) * 2 - 1 # -1 to 1
    origs= rloc + [0,0,0]
    rdir = np.random.random( (10, 2) ) * 2 - 1 # -1 to 1
    sine = sqrt(1-rdir[:,0]**2)
    xdir = sine * np.cos(rdir[:,1]* np.pi)
    ydir = sine * np.sin(rdir[:,1]* np.pi)
    zdir = rdir[:,0]

    key=''
    for orig,px,py,pz in zip(origs,xdir,ydir,zdir):
        ph = Photon(x=orig, dir=arr([px,py,pz]), mfp= float(opts.mfp) )
        # Propagate
        while ( ph.alive ):
            ph.propagate(plist, verbose=opts.verbose)
        #print ph.vertices
        print 'pathlength = ', ph.pathlength
        print 'n_reflects = ', ph.n_reflects

        fig = plt.figure(figsize=(8,12))
        # Draw plane edges
        ax = fig.add_subplot(111, projection='3d')
        for p in plist:
            pts= np.concatenate((p.corners,[p.corners[0]]))
            ax.plot(pts[:,0],pts[:,1],pts[:,2], color='b')

        # Draw photon path
        pts = arr(ph.vertices)
        ax.plot(pts[:,0],pts[:,1],pts[:,2], 'g-o', ms=2, mec='g')
        # start and end point
        ax.scatter(pts[0,0],pts[0,1],pts[0,2], color='red')
        ax.scatter(pts[-1,0],pts[-1,1],pts[-1,2], color='black')

        ax.set_xlim(-3,3)
        ax.set_ylim(-3,3)
        ax.set_zlim(-6,6)
            
        plt.show()

        key= raw_input('q to quit')
        if ( key == 'q' ):
            break
        else:
            plt.clf()
            plt.close()

####------------------------------------------------
if __name__ == '__main__':
    main()
