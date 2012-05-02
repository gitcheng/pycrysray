import numpy as np
import sys
arr = np.array
sqrt = np.sqrt
exp = np.exp
sin = np.sin
cos = np.cos

## Functions ===========================================
def norm_vect(v):
    n = v.dot(v)
    if ( n > 0 ):
        return v / sqrt(n)
    return v

def vect_norm(v):
    return sqrt( v.dot(v) )

def distance(x1,x2):
    return vect_norm( x2-x1 )

def sine_angle(v1,v2,vtx=None):
    '''
    sine of the angle formed by either three points v1, v2, and vtx with vtx as the 
    vertex, or angle between two vectors v1 and v2.
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
def rect_prism(center,dim, random_reflect, diffuse_reflect, diffuse_sigma ):
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
    def __init__(self, x=None, dir=None, wavelength=None, mfp=1e9):
        zero3= np.zeros(3)
        vx = arr([1.0,0.0,0.0], dtype='Float64')
        if ( x == None ):  self.x = zero3
        else: self.x = arr(x, dtype='Float64')
        if ( dir == None ):  self.dir = vx
        else: self.dir = norm_vect(arr(dir))
        if ( wavelength == None ): self.wavelength = 400e-9   # in meter
        else: self.wavelength = wavelength
        self.alive = True
        self.pathlength = 0.0
        self.n_reflects = 0
        self.vertices= [x]
        self.mother_box= arr([10]*3)
        self.mfp= mfp

    def advance(self, d):
        '''
        Advance a distance d.
        '''
        prob = exp( -d/self.mfp )
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

    def reflect(self, plane=None, normal=None, random_reflect=0, diffuse_reflect=0, diffuse_sigma=0):
        '''
        Change direction as it reflects at a surface and add this position to 
        the list of vertices. If a plane is given, normal and reflectivity
        will be taken from the plane. Arguments will be ignored.

        random_reflect: probability of random reflection
        diffuse_reflect: probability of diffuse reflection
        diffuse_sigma: width (sigma) in degrees of diffuse reflection
        '''
        if ( plane==None and normal==None ):
            raise TypeError('Need either plane or normal')

        rndrf= random_reflect
        difrf= diffuse_reflect
        difsig= diffuse_sigma * np.pi/180.0
        if ( plane!= None):
            normal= plane.normal
            rndrf= plane.random_reflect
            difrf= plane.diffuse_reflect
            difsig= plane.diffuse_sigma * np.pi/180.0

        normal = norm_vect(normal)

        # normal must point inward, i.e., if the photon direction is in the
        # same hemisphere as the normal, flip normal
        if ( normal.dot( self.dir ) > 0 ):
            normal = -normal

        self.vertices.append( self.x )
        self.dir = norm_vect(self.dir)

        # Survive or not
        rn = np.random.random_sample()
        if ( rn >= rndrf+difrf ): # not reflected
            self.alive = False

        elif ( rn >= difrf ):   # random (not diffuse)
            xphi = np.random.uniform(-np.pi, np.pi)
            xcth = np.random.uniform(0,0.999)   # cos theta
            xsth = sqrt(1-xcth**2)
            # random vector in upper hemisphere
            vrand = arr([xsth*cos(xphi), xsth*sin(xphi), xcth])
            # rotate the vector so that its z axis points to normal
            self.dir = rotate_vector( vrand, normal)

        else: # diffuse
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
            pass
        
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
        Return the distance and the plane that this photon will hit the first
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

    def propagate(self, plist, sens_boxes=None, verbose=False):
        '''
        Propagating this photon (A recursive function).
        Return True if it hits a sensitive box
        '''
        if ( verbose ):
            print 'propagate, from point ', self.x, 'at direction', self.dir
        if ( not self.alive ):
            if ( verbose ): print 'Dead on arrival'
            return False
        if ( abs(self.x[0])>self.mother_box[0] or
             abs(self.x[1])>self.mother_box[1] or
             abs(self.x[2])>self.mother_box[2] ):
            self.alive=False
            return False

        # find the next plane
        imin,dmin= self.nearest_plane(plist)
        if ( imin == None ):  # did not find any plane
            self.alive = False
            if ( verbose ): print 'Cannot find next plane'
            return False
        if ( verbose ):
            print 'path to next plane is', dmin
            plist[imin].print_prop()
        # advance photon
        self.advance(dmin)
        if ( not self.alive ): return False
        # check if it hits sensitive boxes
        if ( sens_boxes != None ):
            for box in sens_boxes:
                
                return True

        # reflect on this plane
        self.reflect(plist[imin])
        if ( not self.alive ): return False
        # Recursive
        return self.propagate(plist, sens_boxes, verbose)

###################################################################
class Plane:
    '''
    A class describing a flat plane in 3-dim space, defined by an array of 
    corners (3-vector). Only convex polygons are allowed.
    
    corners : an array of space points (arrays of three floats)
    random_reflect : probability of random reflection (angle uniform over (-90,90) degrees.)
    diffuse_reflect : probability of diffuse reflection (angle in Gaussian distribution)
    
    diffuse_sigma : the sigma of the Gaussian.
    '''
    def __init__(self, corners, random_reflect=0.10, diffuse_reflect=0.90, diffuse_sigma=22 ):

        shape = corners.shape
        if ( shape[0]<3 or shape[1]!=3 ): # corners is a an array of shape (n,3), n>=3
            raise TypeError('corners in %s(corners) must be of shape (n,3) with n>=3. But it is (%d,%d)' % ( self.__class__.__name__, shape[0], shape[1])) 
        self.corners = corners
        # Use the two edges at corners[0] to define its normal vector
        self.normal= norm_vect( np.cross(corners[-1]-corners[0], corners[1]-corners[0]) )
        # Center of gravity
        self.cog = np.average( corners, axis=0 )
        self.tolerance= 1e-6

        # Reflectivity
        #  probability of random reflection (angle uniform over (-90,90) degrees.)
        self.random_reflect = random_reflect
        #  probability of diffuse reflection (angle in Gaussian distribution
        #   within sigma = diffuse_sigma 
        self.diffuse_reflect = diffuse_reflect
        self.diffuse_sigma = diffuse_sigma

        # base vectors
        self.u = norm_vect( corners[0]- self.cog )
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
        '''
        return a*self.u+b*self.v

    def print_prop(self):
        print 'Plane:'
        print '  corners:'
        for c in self.corners:
            print c
        print '  normal =', self.normal
        print '  reflectivity =', self.reflectivity

    def on_plane(self, x):
        if ( abs( (x-self.corners[0]).dot( self.normal ) ) > self.tolerance ):
            return False
        else:
            return True
    
    def in_polygon(self, x):
        '''
        Return True if the position x is inside the polygon
        '''
        if ( not self.on_plane(x) ): return False
        nrm = np.cross( self.corners[-1]-x, self.corners[0]-x )
        for c1,c2 in zip(self.corners[:-1], self.corners[1:]):
            nn = np.cross( c1-x, c2-x )
            if ( nrm.dot( nn ) < 0 ):
                return False
                
        return True


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

