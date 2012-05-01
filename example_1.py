#!/usr/bin/env python
import numpy as np
import sys
from pycrysray import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
arr = np.array


def main():

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('--seed', dest='seed', default=12345, help='Random seed')
    parser.add_option('--refidx', dest='refidx', default=0.9, help='Reflectivity')
    parser.add_option('--mfp', dest='mfp', default=1000, help='Mean free path')
    parser.add_option('--verbose', dest='verbose', action='store_true', default=False)
    (opts, args) = parser.parse_args()
    
    ## Random number seed
    np.random.seed(int(opts.seed))

    # Define crystal planes with corners
    #  Dimensions
    center= arr([0,0,0])
    dims = arr([1.5, 1.5, 5.5])  # half lengths
    # Create planes
    ref= float(opts.refidx)
    plist = rect_prism(center, dims, ref)

    # Create photons
    rloc = np.random.random( (10, 3) ) * 2 - 1 # -1 to 1
    origs= rloc * dims*0.99 + center
    rdir = np.random.random( (10, 2) ) * 2 - 1 # -1 to 1
    sine = sqrt(1-rdir[:,0]**2)
    xdir = sine * np.cos(rdir[:,1]* np.pi)
    ydir = sine * np.sin(rdir[:,1]* np.pi)
    zdir = rdir[:,0]

    key=''
    for orig,px,py,pz in zip(origs,xdir,ydir,zdir):
        ph = Photon(x=orig, dir=arr([px,py,pz]), mfp= float(opts.mfp) )
        # Propagate
        ph.propagate(plist, opts.verbose)
        if ( opts.verbose ):
            print ph.vertices
        print 'pathlength = ', ph.pathlength
        print 'n_reflects = ', ph.n_reflects

        # Draw plane edges
        fig = plt.figure(figsize=(5,8))
        ax = fig.add_subplot(111, projection='3d')
        for p in plist:
            pts= np.concatenate((p.corners,[p.corners[0]]))
            ax.plot(pts[:,0],pts[:,1],pts[:,2], color='b')

        # Draw photon path
        pts = arr(ph.vertices)
        ax.plot(pts[:,0],pts[:,1],pts[:,2], color='g')
        
        ax.set_xlim(-3,3)
        ax.set_ylim(-3,3)
        ax.set_zlim(-6,6)
            
        plt.show()

        key= raw_input()
        if ( key == 'x' ):
            break
        else:
            plt.clf()

####------------------------------------------------
if __name__ == '__main__':
    main()
