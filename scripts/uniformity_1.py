#!/usr/bin/env python
'''
Light collection uniformity study
'''
import numpy as np
import matplotlib.pyplot as plt
from pycrysray import *
arr = np.array

def main():

    from optparse import OptionParser
    parser = OptionParser(__doc__)

    parser.add_option('--seed', dest='seed', default=31416, help='Random seed')
    parser.add_option('--refidx', dest='refidx', default='0.0:0.99:0.0',\
                          help='Reflectivity properties: three parameters are needed, random_reflect_probability:diffuse_reflect_probability:diffuse_sigma(degree)')
    parser.add_option('--mfp', dest='mfp', default=1000, help='Mean free path')
    parser.add_option('-n','--n_photons', dest='n_photons', default=1, \
                          help='Number of photons to generate')
    parser.add_option('--length', dest='length', default=11,\
                          help='Crystal length')
    parser.add_option('--apdx', dest='apdx', default=0.5, help='Sensor size in x')
    parser.add_option('--apdy', dest='apdy', default=1.0, help='Sensor size in y')
    
    parser.add_option('--save', dest='save', default='', help='Save result to this file')
    parser.add_option('--draw', dest='draw', action='store_true',default=False,
                      help='Draw')
    (opts, args) = parser.parse_args()


    print opts

    ## Random number seed
    np.random.seed(int(opts.seed))

    ## Geometry
    length = float(opts.length)
    center= arr([0,0, length/2.0])
    dims = arr([3,3,length])/2.0
    # Surface properties
    refpars = opts.refidx.split(':')
    ref= {'random_reflect': float(refpars[0]),
          'diffuse_reflect':float(refpars[1]),
          'diffuse_sigma':float(refpars[2])} 

    plist = rect_prism(center, dims, **ref)

    # Sensor box
    dx = float(opts.apdx) / 2
    dy = float(opts.apdy) / 2
    z= 0.0
    p_apd1= arr([[0.5-dx,dy,z],[0.5+dx,dy,z],[0.5+dx,-dy,z],[0.5-dx,-dy,z]])
    p_apd2= arr([[-0.5-dx,dy,z],[-0.5+dx,dy,z],[-0.5+dx,-dy,z],[-0.5-dx,-dy,z]])

    # draw boxes
    if ( opts.draw ):
        fig0 = plt.figure(figsize=(6,6))
        plt.plot(p_apd1[:,0],p_apd1[:,1])
        plt.plot(p_apd2[:,0],p_apd2[:,1])
        plt.xlim(-1.5,1.5)
        plt.ylim(-1.5,1.5)
        plt.show()

    sensor= {'random_reflect' : 0.0,
             'diffuse_reflect': 0.0,
             'diffuse_sigma'  : 0.0,
             'sensitive' : True} 
    apd1= Plane(p_apd1, **sensor)
    apd2= Plane(p_apd2, **sensor)
    sensors = [apd1, apd2]

    nphs= int(opts.n_photons)   # number of photons
    mfp = float(opts.mfp)   # mean free path
    posz = np.arange( center[2]-length/2+1, center[2]+length/2, 1.0 )
    eff = []     # efficiency
    for z in posz:
        # photon origin location. Z is fixed, (x,y) randomized over a square
        ox = np.random.random( nphs ) *2 -1   # -1,+1
        oy = np.random.random( nphs ) *2 -1   # -1,+1
        oz = np.ones( nphs )*z
        orig = arr([ox,oy,oz]).transpose()
        # randomize angles
        cth = np.random.uniform(-1,1, nphs)
        phi = np.random.uniform(-np.pi,np.pi, nphs)
        sth = np.sqrt(1-cth**2)
        xdir = sth * np.cos(phi)
        ydir = sth * np.sin(phi)
        zdir = cth
        dir = arr([xdir, ydir, zdir]).transpose()
        # create photons
        nhits= 0
        for oo, dd in zip(orig,dir):
            ph = Photon(x=oo, dir=dd, mfp=mfp)
            hit= None
            while ( ph.alive ):
                hit= ph.propagate(plist, black_regions=sensors)
            if ( hit!=None):
                if ( hit.sensitive ):
                    nhits= nhits+1
        eff.append( float(nhits)/float(nphs) )

    # Print results
    eff= arr( eff )
    err = np.sqrt( eff*(1-eff) / nphs )
    result = arr([posz, eff,err]).transpose()
    print result

    if ( opts.save != '' ):
        np.savez(opts.save, data=result, meta=opts)



###########
if __name__ == '__main__':
    main()
