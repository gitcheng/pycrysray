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
    parser.add_option('--draw', action='store_true', dest='draw', default=False,
                      help='Draw picture')
    parser.add_option('--seed', dest='seed', default=12345, help='Random seed')
    parser.add_option('--refidx', dest='refidx', default=0.9, help='Reflectivity')
    parser.add_option('--mfp', dest='mfp', default=1000, help='Mean free path')
    parser.add_option('--verbose', dest='verbose', action='store_true', default=False)
    parser.add_option('-n','--n_photons', dest='n_photons', default=1, \
                          help='Number of photons to generate')
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
    rloc = np.random.random( (int(opts.n_photons), 3) ) * 2 - 1 # -1 to 1
    rdir = np.random.random( (int(opts.n_photons), 2) ) * 2 - 1 # -1 to 1
    origs= rloc * [3,3,3] + [0,0,0]
    sine = sqrt(1-rdir[:,0]**2)
    xdir = sine * np.cos(rdir[:,1]* np.pi)
    ydir = sine * np.sin(rdir[:,1]* np.pi)
    zdir = rdir[:,0]

    # some histograms
    h_plen= arr([0.0]*len(rloc))
    h_nflx= arr([0.0]*len(rloc))

    for j,(orig,px,py,pz) in enumerate(zip(origs,xdir,ydir,zdir)):
        ph = Photon(x=orig, dir=arr([px,py,pz]), mfp= float(opts.mfp) )
        # Propagate
        ph.propagate(plist, opts.verbose)
        h_plen[j] = ph.pathlength
        h_nflx[j] = ph.n_reflects
        #print 'pathlength = ', ph.pathlength
        #print 'n_reflects = ', ph.n_reflects

    if ( opts.draw ):
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(211)
        ax.semilogx()
        ax.hist(h_plen,100)
        ax = fig.add_subplot(212)
        ax.hist(h_nflx,100)
        plt.show()


####------------------------------------------------
if __name__ == '__main__':
    main()
