import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
arr = np.array
sqrt = np.sqrt
exp = np.exp
sin = np.sin
cos = np.cos

## Functions ===========================================
def unit_vect(v):
    '''
    Return the unit vector of direction v.
    '''
    n = v.dot(v)
    if ( n > 0 ):
        return v / sqrt(n)
    return v

def vect_norm(v):
    '''
    Return the norm of a vector |v|
    '''
    return sqrt( v.dot(v) )

def distance(x1,x2):
    '''
    Return the distance between two points |x1-x2|
    '''
    return vect_norm( x2-x1 )

def sine_angle(v1,v2,vtx=None):
    '''
    sine of the angle formed by either three points v1, v2, and vtx with vtx as
    the vertex, or angle between two vectors v1 and v2 if vtx=None.
    '''
    a,b = v1,v2
    if ( vtx != None ):
        a= a-vtx
        b= b-vtx
    n= np.cross(a,b) / ( vect_norm(a) * vect_norm(b) )
    return vect_norm(n)

def rotate_vector(v, zhat):
    '''
    Rotate the vector v so that the z axis of its coordinate points to zhat.
    '''
    # Get the angles of zhat
    abszhat = vect_norm(zhat)
    if ( abszhat == 0 ):
        raise TypeError('zhat is zero!!')
    ctheta = zhat[2]/abszhat
    stheta = sqrt(1-ctheta**2)
    cphi, sphi= 1, 0
    if ( stheta > 1e-6 ):
        cphi = zhat[0]/ (abszhat * stheta)
        sphi = zhat[1]/ (abszhat * stheta)
    # rotate matrix, about y-axis, by angle theta
    Ry = np.matrix([ [ctheta, 0, stheta], [0,1,0], [-stheta, 0, ctheta] ] )
    # about z-axis, by angle phi
    Rz = np.matrix([ [cphi,-sphi,0],[sphi,cphi,0],[0,0,1] ] )
    # Rotate v
    return arr(Rz*Ry* v.reshape(3,1)).reshape(3,)

## Geometries ===================================================
def rect_prism(center, dim, random_reflect, diffuse_reflect, diffuse_sigma ):
    '''
    Return a rectangular prism whose edges are parallel to the axes.
    *center* is the coordinate of the center of gravity
    *dim* is half length of the three edges
    '''
    cns=[]
    for i in range(-1,2,2):
        for j in range(-1,2,2):
            for k in range(-1,2,2):
                cns.append( center+dim*[i,j,k] )
    # Four corners of each plane
    cs1 = arr([ cns[0], cns[1], cns[3], cns[2] ])
    cs2 = arr([ cns[4], cns[5], cns[7], cns[6] ])
    cs3 = arr([ cns[0], cns[2], cns[6], cns[4] ])
    cs4 = arr([ cns[0], cns[1], cns[5], cns[4] ])
    cs5 = arr([ cns[2], cns[3], cns[7], cns[6] ])
    cs6 = arr([ cns[1], cns[3], cns[7], cns[5] ])

    ref= {'random_reflect':random_reflect, 'diffuse_reflect':diffuse_reflect, 'diffuse_sigma':diffuse_sigma }
    plist = [Plane(cs1,**ref),Plane(cs2,**ref),Plane(cs3,**ref),Plane(cs4,**ref),Plane(cs5,**ref),Plane(cs6,**ref)]
    return plist

###################################################################
class Photon:
    '''
    A class describing a photon
    x : Current position (array of three floats)
    p : Current direction (array of three floats, norm=1
    wavelength : wavelength
    mfp : Mean free path
    '''
    def __init__(self, x, dir, wavelength=None, mfp=1e9):
        zero3= np.zeros(3)
        self.x = arr(x, dtype='Float64')
        self.dir = unit_vect(arr(dir))
        if ( wavelength == None ): self.wavelength = 400e-9   # in meter
        else: self.wavelength = wavelength
        self.alive = True
        self.detected = False
        self.pathlength = 0.0
        self.n_reflects = 0
        self.vertices= [x]
        self.mother_box= arr([100.,100.,100.])
        self.mfp= mfp
        self.deposit_at= None   # At which sensor

    def advance(self, d):
        '''
        Advance a distance d.
        '''
        prob = exp( -abs(d)/self.mfp )
        if ( np.random.uniform() > prob ):
            self.alive = False

        if ( self.alive ):
            self.x = self.path(d)
            self.pathlength += d

    def path(self, d):
        '''
        A function that returns the coordinate on the photon path
        '''
        return self.x + d* self.dir

    def reflect(self, plane, sensors=None):
        '''
        Change direction as it reflects at a surface and add this position to 
        the list of vertices.
        Return the plane the photon is on.
        *plane*: The plane the photon is reflecting on
        *sensors*: a list of Plane object that represents the sensors
        '''
        ## Surface property
        normal= plane.normal
        rndrf= plane.random_reflect
        difrf= plane.diffuse_reflect
        difsig= plane.diffuse_sigma * np.pi/180.0
        ifr_in= plane.idx_refract_in
        ifr_out= plane.idx_refract_out

        insensor= None
        for sp in sensors:
            # If the photon is inside the sensor area, change surface property
            if sp.contain( self.x ):
                insensor= sp
                normal= sp.normal
                rndrf= sp.random_reflect
                difrf= sp.diffuse_reflect
                difsig= sp.diffuse_sigma * np.pi/180.0
                ifr_in= sp.idx_refract_in
                ifr_out= sp.idx_refract_out
                break

        normal = unit_vect(normal)
        # normal must point inward, i.e., if the photon direction is in the
        # same hemisphere as the normal, flip normal
        if ( normal.dot( self.dir ) > 0 ):
            normal = -normal

        self.vertices.append( self.x )
        self.dir = unit_vect(self.dir)

        # Total reflection or not?
        costh= normal.dot(self.dir)
        sinth= np.sqrt(1- costh**2)
        randomed= False
        diffused= False
        if sinth > ifr_out/float(ifr_in):  ## total reflection
            randomed= False
            diffused= True
        else:
            rn= np.random.random_sample()
            if rn >= rndrf+difrf: # not reflected
                self.alive = False
                randomed= False
                diffused= False
            elif rn >= difrf:  #  random reflection
                randomed= True
                diffused= False
            else:    # diffused reflection
                randomed= False
                diffused= True

        if not self.alive and insensor is not None:
            self.deposit_at= insensor

        # How is it reflected
        if self.alive and randomed:   # random reflection
            xphi = np.random.uniform(-np.pi, np.pi)
            xcth = np.random.uniform(0,0.999)   # cos theta
            xsth = sqrt(1-xcth**2)
            # random vector in upper hemisphere
            vrand = arr([xsth*cos(xphi), xsth*sin(xphi), xcth])
            # rotate the vector so that its z axis points to normal
            self.dir = rotate_vector( vrand, normal)

        if self.alive and diffused: # diffuse
            # specular reflection
            spec_dir = self.dir - 2*self.dir.dot(normal) * normal
            while ( True ):
                # random phi
                xphi = np.random.uniform(-np.pi, np.pi)
                # Gaussian theta
                xtheta = np.random.randn() * difsig
                xsth= sin(xtheta)
                xcth= cos(xtheta)
                # random vector in upper hemisphere, around z-axis
                vrand = arr([xsth*cos(xphi), xsth*sin(xphi), xcth])
                # rotate the vector so that its z axis points to specular reflection
                vrand = rotate_vector( vrand, spec_dir)
                # If vrand is less than 90 degrees of normal, ok, break
                if ( vrand.dot(normal) > 1e-6 ):
                    self.dir = vrand
                    break
        
        if ( self.alive ):
            self.n_reflects += 1
            # to stay away from this plane. FIXME!! This will make the edges leak.
            self.advance(1e-6)

    def path_to(self, aplane):
        '''
        Pathlength (signed) to a plane.
        '''
        dom = self.dir.dot(aplane.normal)
        return aplane.normal.dot(aplane.corners[0] - self.x) / dom

    def nearest_plane(self, plist):
        '''
        Return the index of the next plane and the distance to that plane
        (index, distance)
        '''
        dmin = 1e9
        imin = None
        for i in xrange(len(plist)):
            d = self.path_to(plist[i])
            if ( d < 0 ): continue
            if d < dmin:
                dmin = d
                imin = i
        return imin, dmin

    def propagate(self, crystal, verbose=False):
        '''
        Propagating this photon (A recursive function).
        Return the plane this photon dies on.
        '''
        if ( verbose ):
            print 'propagate, from point ', self.x, 'at direction', self.dir
        if ( not self.alive ):
            if ( verbose ): print 'Dead on arrival'
            return None
        if ( abs(self.x[0])>self.mother_box[0] or
             abs(self.x[1])>self.mother_box[1] or
             abs(self.x[2])>self.mother_box[2] ):
            self.alive=False
            return None

        # find the next plane
        imin,dmin= self.nearest_plane(crystal.planes)
        if ( imin == None ):  # did not find any plane
            self.alive = False
            if ( verbose ): print 'Cannot find next plane'
            return None
        if ( verbose ):
            print 'path to next plane is', dmin
            crystal.planes[imin].print_properties()
        # advance photon
        self.advance(dmin)
        if ( not self.alive ): return None

        # reflect on this plane
        self.reflect(crystal.planes[imin], crystal.sensors)

        if not self.alive:
            return crystal.planes[imin]

        return self.propagate(crystal, verbose)



###################################################################
class Plane:
    '''
    A class describing a flat plane in 3-dim space, defined by an array of 
    corners (3-vector). Only convex polygons are allowed.
    
    *corners* : an array of space points (arrays of three floats). So the shape
                of corners must be (n,3),  n>=3
    *random_reflect* : probability of random reflection (angle uniform over 
        (-90,90) degrees.)
    *diffuse_reflect* : probability of diffuse reflection (angle in Gaussian 
        distribution)
    *diffuse_sigma* : the sigma of the Gaussian.
    *idx_refract_in* : index of refraction on this side of the plane
    *idx_refract_out* : index of refraction on the other side of the plane
    *sensitive* : Is this plane a sensitive object (photo sensor) or not.
    '''
    def __init__(self, corners, random_reflect=0.10, diffuse_reflect=0.90, diffuse_sigma=22, idx_refract_in=1, idx_refract_out=1, sensitive=False ):

        shape = corners.shape
        if ( shape[0]<3 or shape[1]!=3 ): # corners is a an array of shape (n,3), n>=3
            raise TypeError('corners in %s(corners) must be of shape (n,3) with n>=3. But it is (%d,%d)' % ( self.__class__.__name__, shape[0], shape[1])) 
        self.corners = corners

        # Use the two edges at corners[0] to define its normal vector
        self.normal= unit_vect( np.cross(corners[-1]-corners[0], corners[1]-corners[0]) )
        # Center of gravity
        self.cog = np.average( corners, axis=0 )
        self.tolerance= 1e-6
        # Sensitive surface or not
        self.sensitive= sensitive

        # Reflectivity
        #  probability of random reflection (angle uniform over (-90,90) degrees.)
        self.random_reflect = random_reflect
        #  probability of diffuse reflection (angle in Gaussian distribution
        #   within sigma = diffuse_sigma 
        self.diffuse_reflect = diffuse_reflect
        self.diffuse_sigma = diffuse_sigma
        #  Index of reflection
        self.idx_refract_in = idx_refract_in
        self.idx_refract_out = idx_refract_out

        # base vectors
        self.u = unit_vect( corners[0]- self.cog )
        self.v = np.cross( self.normal, self.u )
        ## Check none of the corners are too close
        for i in xrange(shape[0]-1):
            for j in range(i+1,shape[0]):
                if ( vect_norm( corners[i]-corners[j] ) < self.tolerance ):
                    print corners[i], corners[j]
                    raise TypeError('Two corners shown above are the same or too close.')
        
        ## Check no three consecutive corners are on a straight line
        for i in xrange( shape[0] ):
            if ( sine_angle(corners[i-2],corners[i-1],corners[i]) < self.tolerance ) :
                print corners[i-2],corners[i-1],corners[i]
                raise TypeError('The corners shown above are on a straight line.')
        
        ## Check all its interior angles are less than 180, or cross product
        ## does not flip direction
        for i in xrange( shape[0] ):
            n = np.cross( corners[i-2]-corners[i-1], corners[i]-corners[i-1] )
            if ( n.dot( self.normal ) < 0 ) :
                print corners[i-1], corners[i], corners[i+1]
                raise TypeError( 'These corners form an angle that is greater than 180 degrees. Only convex polygons are allowed.')

        ## Check corners are on the same plane
        for c in corners:
            if ( abs( (c-corners[0]).dot( self.normal ) ) > self.tolerance ):
                print c
                raise TypeError('The corner shown above is not on the same plane')

    def plane_func(self, a, b):
        '''
        Return a point on this plane given the parameters (a,b), using
        the first and the last edges for the basis.
        p = a*u + b*v
        '''
        return a*self.u+b*self.v

    def print_properties(self):
        '''
        Print out the property of this plane
        '''
        print 'Plane:'
        print '  corners:'
        for c in self.corners:
            print c
        print '  normal =', self.normal
        print '  random_reflect =', self.random_reflect
        print '  diffuse_reflect =', self.diffuse_reflect
        print '  diffuse_sigma =', self.diffuse_sigma
        print '  index of refraction in = ', self.idx_refract_in
        print '  index of refraction out = ', self.idx_refract_out

    def on_plane(self, x):
        '''
        True if x is on the 2-D plane (infinite)
        '''
        if ( abs( (x-self.corners[0]).dot( self.normal ) ) > self.tolerance ):
            return False
        else:
            return True
    
    def contain(self, x):
        '''
        Return True if the position x is inside the 2D polygon
        '''
        if ( not self.on_plane(x) ): return False
        nrm = np.cross( self.corners[-1]-x, self.corners[0]-x )
        for c1,c2 in zip(self.corners[:-1], self.corners[1:]):
            nn = np.cross( c1-x, c2-x )
            if ( nrm.dot( nn ) < 0 ):
                return False
                
        return True



###################################################################
class Crystal:
    '''
    Class crystal. Defined by a list of Planes, and a list of sensors
    Ex.
       crystal_1= Crystal([p1, p2, p3, p4, p5, p6], [s1, s2])
        pi and si are instances of Plane
    '''
    def __init__(self, planes, sensors):
        self.planes= planes
        self.sensors= sensors

    def draw(self, ax=None):
        '''
        Draw the crystal in 3D axis
        '''
        if ax is None:
            ax= plt.gca()
        for p in list(self.planes)+list(self.sensors):
            pts= np.concatenate((p.corners,[p.corners[0]]))
            color='b'
            if p.sensitive: color='r'
            ax.plot(pts[:,0],pts[:,1],pts[:,2], color=color)


###################################################################

if __name__ == '__main__':

    p = Photon(x=[1,2,3], dir=[1,-1,0])
    print p.x, p.x.dtype
    print p.dir
    print p.wavelength
    olddir= p.dir

    p.reflect([0,1,0])
    print 'newdir = ', p.dir
    print 'olddir = ', olddir
    print p.dir-olddir
    print p.n_reflects

    corners = arr( [ [0,0,0], [2,0,0], [1,2,0], [0,1,0] ] )
    s1 = Plane(corners)
    print s1.normal
    print s1.plane_func(10,20)


    print 'x = ', p.x, ' dir= ', p.dir
    print p.path(10)
    p.advance(10)
    print 'x = ', p.x

