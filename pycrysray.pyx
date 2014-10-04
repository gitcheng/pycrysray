cimport cython
import numpy as np
cimport numpy as np
nprnd= np.random
from cpython cimport bool
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from math import sin as msin
from math import cos as mcos
from math import sqrt as msqrt
from math import exp as mexp
from math import radians as mradians
from scipy.constants import c as clight
clightcm = clight * 100
clightcmns = clightcm * 1e-9  ## speed of light in cm/ns
npsqrt= np.sqrt
npexp= np.exp
npsin= np.sin
npcos= np.cos

DTYPE = np.float
ctypedef np.float64_t DTYPE_t

## Functions ===========================================
@cython.boundscheck(False)
@cython.wraparound(False)
def vect_norm(DTYPE_t[::1] vec):
    '''
    Return the norm of a vector |v|
    '''
    return msqrt( vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2])
    #cdef DTYPE_t norm= 0.0
    #for v in vec:
    #    norm+= v*v
    #return msqrt(norm)

def unit_vect(np.ndarray[DTYPE_t, ndim=1] v):
    '''
    Return the unit vector of direction v.
    '''
    return v*(1.0/vect_norm(v))

def distance(np.ndarray[DTYPE_t, ndim=1] x1, np.ndarray[DTYPE_t, ndim=1] x2, int n=3):
    '''
    Return the distance between two points |x1-x2|
    '''
    return vect_norm( x2-x1 )

def sine_angle(np.ndarray[DTYPE_t, ndim=1] v1, np.ndarray[DTYPE_t, ndim=1] v2, np.ndarray[DTYPE_t, ndim=1] vtx=None):
    '''
    sine of the angle formed by either three points v1, v2, and vtx with vtx as
    the vertex, or angle between two vectors v1 and v2 if vtx=None.
    '''
    cdef np.ndarray[DTYPE_t, ndim=1] a= v1
    cdef np.ndarray[DTYPE_t, ndim=1] b= v2

    if ( vtx != None ):
        a= a-vtx
        b= b-vtx
    cdef np.ndarray[DTYPE_t, ndim=1] n= np.cross(a,b) / ( vect_norm(a) * vect_norm(b) )
    return vect_norm(n)

@cython.boundscheck(False)
def rotate_vector(np.ndarray[DTYPE_t, ndim=1] v, np.ndarray[DTYPE_t, ndim=1] zhat):
    '''
    Rotate the vector v so that the z axis of its coordinate points to zhat.
    '''
    # Get the angles of zhat
    cdef DTYPE_t abszhat, ctheta, stheta, cphi, sphi
    abszhat = vect_norm(zhat)
    if ( abszhat == 0 ):
        raise TypeError('zhat is zero!!')
    ctheta = zhat[2]/abszhat
    stheta = msqrt(1-ctheta**2)
    cphi, sphi= 1, 0
    if ( stheta > 1e-6 ):
        cphi = zhat[0]/ (abszhat * stheta)
        sphi = zhat[1]/ (abszhat * stheta)
    # rotate matrix, about y-axis, by angle theta
    cdef np.ndarray[DTYPE_t, ndim=2] Ry, Rz
    Ry = np.matrix([ [ctheta, 0.0, stheta], [0.0,1.0,0.0], [-stheta, 0.0, ctheta] ] )
    # about z-axis, by angle phi
    Rz = np.matrix([ [cphi,-sphi,0.0],[sphi,cphi,0.0],[0.0,0.0,1.0] ] )
    # Rotate v
    return np.array(Rz*Ry* v.reshape(3,1)).reshape(3,)

## Geometries ===================================================
def rectangle(center, xlen, ylen, **kwargs):
    '''
    Return a rectangular Plane centered at center (array of three floats)
    with xlength and ylength
    '''
    x1= center[0] - xlen/2.
    x2= center[0] + xlen/2.
    y1= center[1] - ylen/2.
    y2= center[1] + ylen/2.
    z= center[2]
    return Plane([[x1,y1,z],[x1,y2,z],[x2,y2,z],[x2,y1,z]], **kwargs)

def hexagon(center, width, **kwargs):
    '''
    Return a 2D hexagon Plane on x-y plane. 
    *width*: Side-to-side distance
    '''
    cosphs= np.cos(np.arange(6)*np.pi/3.0)
    sinphs= np.sin(np.arange(6)*np.pi/3.0)
    cx= cosphs* width/np.sqrt(3) + center[0]
    cy= sinphs* width/np.sqrt(3) + center[1]
    cz= center[2]*np.ones(6)
    cs1= np.array( zip(cx, cy, cz) )
    return Plane(cs1, **kwargs)
    

def ngon_taper(plane1, plane2, **kwargs):
    '''
    Return a tapered (or straight, depending on the size of the two end-planes) 
    prism.
    '''
    if len(plane1.corners)!=len(plane2.corners):
        raise ValueError('The two end-planes have different number of sides')
    ncorners= len(plane1.corners)
    cn1, cn2= plane1.corners, plane2.corners
    plist= [plane1, plane2]
    for i in range(ncorners):
        k= np.mod(i+1, ncorners)
        plist.append(Plane([cn1[i], cn1[k], cn2[k], cn2[i]], **kwargs))
    return plist

def rect_taper(center, length, xlen1, ylen1, xlen2, ylen2, **kwargs):
    '''
    Tapered rectangular prism
    '''
    acenter= np.array(center, dtype=np.float)
    c1= acenter- np.array([0,0,length/2.])
    c2= acenter+ np.array([0,0,length/2.])
    rec1= rectangle(c1, xlen1, ylen1, **kwargs)
    rec2= rectangle(c2, xlen2, ylen2, **kwargs)
    return ngon_taper(rec1, rec2, **kwargs)


def rect_prism(center, length, xlen, ylen, **kwargs):
    '''
    Return a rectangular prism whose edges are parallel to the axes.
    *center* is the coordinate (array of three floats) of the center of gravity
    kwargs are plane properties. See class Plane.
    '''
    return rect_taper(center, length, xlen, ylen, xlen, ylen, **kwargs)

def hex_taper(center, length, width1, width2, **kwargs):
    '''
    Return a tapered hexagonal prism
    '''
    acenter= np.array(center, dtype=np.float)
    ct1= acenter+ np.array([0,0,-length/2.])
    ct2= acenter+ np.array([0,0,+length/2.])
    hex1= hexagon(ct1, width1, **kwargs)
    hex2= hexagon(ct2, width2, **kwargs)
    return ngon_taper(hex1, hex2, **kwargs)

def hex_prism(center, length, width, **kwargs): 
    '''
    Return a hexagonal prism whose edges are parallel to the axes.
    *center* is the coordinate (array of three floats) of the center of gravity
    *length* : prism total length in z
    *width* : hexagon size, edge to the opposite edge
    kwargs are plane properties. See class Plane.
    '''
    return hex_taper(center, length, width, width, **kwargs)

# Reflectance and transmittance
# Fresnel equations http://en.wikipedia.org/wiki/Fresnel_equations
# s-polarized and p-polarized
#n1-->n2
def reflectance_s(DTYPE_t n1, DTYPE_t n2, DTYPE_t cos_theta_i):
    cdef DTYPE_t cti= abs(cos_theta_i)
    cdef DTYPE_t sin_theta_i = msqrt(1- cti**2)
    if n1/n2*sin_theta_i > 1:
        return 1
    cdef DTYPE_t cos_theta_t = msqrt(1-(n1/n2*sin_theta_i)**2)
    cdef DTYPE_t ret = (n1*cti - n2* cos_theta_t)/(n1*cti + n2* cos_theta_t)
    ret = ret**2
    return ret
def transmittance_s(DTYPE_t n1, DTYPE_t n2, DTYPE_t cos_theta_i):
    return 1-reflectance_s(n1,n2,cos_theta_i)
def reflectance_p(DTYPE_t n1, DTYPE_t n2, DTYPE_t cos_theta_i):
    cdef DTYPE_t cti= abs(cos_theta_i)
    cdef DTYPE_t sin_theta_i = msqrt(1- cti**2)
    if n1/n2*sin_theta_i > 1:
        return 1.0
    cdef DTYPE_t cos_theta_t = msqrt(1-(n1/n2*sin_theta_i)**2)
    cdef DTYPE_t ret = (n1*cos_theta_t - n2* cti)/(n1*cos_theta_t + n2* cti)
    ret = ret**2
    return ret
def transmittance_p(DTYPE_t n1, DTYPE_t n2, DTYPE_t cos_theta_i):
    return 1.0-reflectance_p(n1,n2,cos_theta_i)

def reflectance(DTYPE_t n1, DTYPE_t n2, DTYPE_t cos_theta_i):
    return 0.5*(reflectance_s(n1,n2,cos_theta_i)+reflectance_p(n1,n2,cos_theta_i))
def transmittance(DTYPE_t n1, DTYPE_t n2, DTYPE_t cos_theta_i):
    return 1.0- reflectance(n1, n2, cos_theta_i)

# Generate random directions and random positions
def generate_p6(np.ndarray[DTYPE_t, ndim=1] center, DTYPE_t dz, DTYPE_t dr):
    '''
    Generate random directions and random positions
    Return two arrays (3-vector), first the position; second the direction.
    *center* is the center of generated positions
    *dz* is the range in z (-dz, +dz)
    *dr* is the range in x-y plane (a circle with radius dr)
    '''
    cdef DTYPE_t x,y,z, r, phi, cth, sth, xdir, ydir 
    z= center[2]+ nprnd.uniform(-1,1)* dz
    r= np.sqrt(nprnd.uniform(0,1)) * dr
    phi= nprnd.uniform(0, 2*np.pi)
    x= center[0] + r*mcos(phi)
    y= center[1] + r*msin(phi)
    #
    phi= nprnd.uniform(0, 2*np.pi)
    cth= nprnd.uniform(-1,1)
    sth= msqrt(1-cth**2)
    xdir= sth*mcos(phi)
    ydir= sth*msin(phi)
    return np.array([x,y,z]), np.array([xdir,ydir,cth])

# Return a random direction in the same hemisphere as normal vector.
def random_direction(normal):
    '''
    Return a random direction in the same hemisphere as normal vector.
    random_direction(normal)
    *normal*: 3-vector (normalized) of the normal
    '''
    xphi = nprnd.uniform(-np.pi, np.pi)
    xcth = nprnd.uniform(0,0.999)   # cos theta
    xsth = msqrt(1-xcth**2)
    # random vector in upper hemisphere
    vrand = np.array([xsth*mcos(xphi), xsth*msin(xphi), xcth])
    # rotate the vector so that its z axis points to normal
    return rotate_vector( vrand, normal)

###################################################################
class Plane:
    '''
    A class describing a flat plane in 3-dim space, defined by an array of 
    corners (3-vector). Only convex polygons are allowed.
    
    *corners* : an array of space points (arrays of three floats). So the shape
                of corners must be (n,3),  n>=3
    *sigdif_crys* : the sigma (degrees) of diffused reflection on crystal surface
    *pdif_crys*: probability of diffused reflection on crystal surface
    *prand_crys*: probability of random reflection on crystal surface
    *sigdif_wrap*: the sigma (degrees) of diffused reflection on wrapper
    *pdif_wrap*: probability of diffused reflection on wrapper
    *prand_wrap*: probability of random reflection on wrapper
    *idx_refract_in* : index of refraction on this side of the plane
    *idx_refract_out* : index of refraction on the other side of the plane
    *sensitive* : Is this plane a sensitive object (photo sensor) or not.
    '''
    def __init__(self, corners, sigdif_crys=0.0, pdif_crys=1.0, prand_crys=0.0,
                 sigdif_wrap=0.0, pdif_wrap=1.0, prand_wrap=0.0,
                 idx_refract_in=1, idx_refract_out=1,
                 sensitive=False, wrapped=True ):

        self.corners = np.array(corners)
        shape = self.corners.shape
        if ( shape[0]<3 or shape[1]!=3 ): # corners is a an array of shape (n,3), n>=3
            raise TypeError('corners in %s(corners) must be of shape (n,3) with n>=3. But it is (%d,%d)' % ( self.__class__.__name__, shape[0], shape[1])) 

        # Use the two edges at corners[0] to define its normal vector
        self.normal= unit_vect( np.cross(self.corners[-1]-self.corners[0],
                                         self.corners[1]-self.corners[0]) )
        # Center of gravity
        self.cog = np.average( self.corners, axis=0 )
        self.tolerance= 1e-7
        # Sensitive surface or not
        self.sensitive= sensitive
        # Reflectivity
        self.sigdif_crys= sigdif_crys
        self.pdif_crys = pdif_crys
        self.prand_crys = prand_crys
        self.sigdif_wrap= sigdif_wrap
        self.pdif_wrap= pdif_wrap
        self.prand_wrap= prand_wrap

        self.wrapped= wrapped
        #  Index of reflection
        self.idx_refract_in = idx_refract_in
        self.idx_refract_out = idx_refract_out
        # base vectors
        self.u = unit_vect( self.corners[0]- self.cog )
        self.v = np.cross( self.normal, self.u )
        #
        self.sanity_check()

    def sanity_check(self):
        shape= self.corners.shape
        ## Check none of the corners are too close
        for i in range(shape[0]-1):
            for j in range(i+1,shape[0]):
                if vect_norm(self.corners[i]-self.corners[j]) < self.tolerance:
                    print self.corners[i], self.corners[j]
                    raise TypeError('Two corners shown above are the same'
                                    ' or too close.')
        
        ## Check no three consecutive corners are on a straight line
        for i in range( shape[0] ):
            if sine_angle(self.corners[i-2],self.corners[i-1],self.corners[i]) < self.tolerance :
                print self.corners[i-2],self.corners[i-1],self.corners[i]
                raise TypeError('The corners shown above are on a straight line.')
        
        ## Check all its interior angles are less than 180, or cross product
        ## does not flip direction
        for i in range( shape[0] ):
            n = np.cross( self.corners[i-2]-self.corners[i-1], self.corners[i]-self.corners[i-1] )
            if n.dot( self.normal ) < 0 :
                print self.corners[i-1], self.corners[i], self.corners[i+1]
                raise TypeError('These corners form an angle that is greater'
                                ' than 180 degrees. Only convex polygons are'
                                ' allowed.')

        ## Check corners are on the same plane
        for c in self.corners:
            if abs( (c-self.corners[0]).dot( self.normal ) ) > self.tolerance:
                print c
                raise TypeError('The corner shown above is not on the same plane')
        
        # Check sensitivity and wrapping
        if self.sensitive and self.wrapped:
            raise TypeError('The surface cannot be sensitive and wrapped')

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
        print '  sigdif_crys =', self.sigdif_crys
        print '  pdif_crys =', self.pdif_crys
        print '  prand_crys =', self.prand_crys
        print '  sigdif_wrap =', self.sigdif_wrap
        print '  pdif_wrap =', self.pdif_wrap
        print '  prand_wrap =', self.prand_wrap
        print '  index of refraction in = ', self.idx_refract_in
        print '  index of refraction out = ', self.idx_refract_out
        print '  sensitive= ', self.sensitive
        print '  wrapped= ', self.wrapped

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

    def normal_side(self, x):
        '''
        Return True if the point is on the same side as the normal.
        '''
        return self.normal.dot(x-self.cog)>0


###################################################################
class Crystal:
    '''
    Class crystal. Defined by a list of Planes, both sensitive and
    insensitive Planes in the same list.
    Ex.
       crystal_1= Crystal('theName', [p1, p2, p3, p4, p5, p6, s1, s2])
        pi and si are instances of Plane
    '''
    def __init__(self, name, planes):
        self.name= name
        self.allplanes= planes
        self.planes= []
        self.sensors= []
        for p in planes:
            if p.sensitive:
                self.sensors.append(p)
            else:
                self.planes.append(p)
        # Center of gravity
        self.cog= np.array( [ p.cog for p in self.planes ] ).mean(axis=0)
        self.__set_plane_normal()

    def set_plane_normal(self):
        '''
        Check if the normals of the planes point inwards. If not, flip it.
        '''
        for p in self.allplanes:
            vc= self.cog - p.cog
            if vc.dot(p.normal) < 0: 
                p.normal= -p.normal

    __set_plane_normal = set_plane_normal

    def draw(self, ax=None, photon=None):
        '''
        Draw the crystal in 3D axis
        '''
        if ax is None:
            ax= plt.gca()
        for p in self.allplanes:
            pts= np.concatenate((p.corners,[p.corners[0]]))
            color='b'
            if p.sensitive: color='orange'
            ax.plot(pts[:,0],pts[:,1],pts[:,2], color=color)

        if photon is not None:
            pts= np.array(photon.vertices)
            ax.plot(pts[:,0],pts[:,1],pts[:,2], 'g-', ms=2, mec='g', alpha=0.5)
            px= photon.startx
            ax.plot([px[0]],[px[1]],[px[2]], 'ro')
            px= photon.x
            ax.plot([px[0]],[px[1]],[px[2]], 'rx', mew=2)

    def contain(self, x):
        '''
        Whether the point x in inside the volume of the crystal
        '''
        for p in self.planes:
            if not p.normal_side(x):
                return False
        return True

###################################################################
class Photon:
    '''
    A class describing a photon
    x : Current position (array of three floats)
    dir : Current direction (array of three floats, norm=1
    t0 : time of creation
    t : current time
    wavelength : wavelength
    mfp : Mean free path
    '''
    def __init__(self, x, dir, t=0, wavelength=None, mfp=1e9,\
                 trackvtx=False):
        self.x = np.array(x, dtype='Float64')
        self.dir = unit_vect(np.array(dir))
        self.startx = x
        self.t0 = t
        self.t = t
        if ( wavelength == None ): self.wavelength = 400e-9   # in meter
        else: self.wavelength = wavelength
        self.alive = True
        self.pathlength = 0.0
        self.n_reflects = 0
        self.incident_costh= -1.0
        self.vertices= [x]
        self.trackvtx= trackvtx
        self.mother_box= np.array([100.,100.,100.])
        self.mfp= mfp
        self.deposit_at= None   # At which sensor
        self.status_names= ['alive','rangeout','absorbed','transmitted','outofvolume','noplane']
        self.status= 0
        self.rangeout= 1
        self.absorbed= 2
        self.transmitted= 3
        self.outofvolume= 4
        self.noplane= 5
        self.lastplane= None

    def advance(self, d, idx_refract=1):
        '''
        Advance a distance d.
        '''
        prob = mexp( -abs(d)/self.mfp )
        if ( nprnd.uniform() > prob ):
            self.alive = False
            self.status = self.rangeout

        if ( self.alive ):
            self.x = self.path(d)
            self.pathlength += d
            self.t += d/clightcmns * idx_refract

    def path(self, d):
        '''
        A function that returns the coordinate on the photon path
        '''
        return self.x + d* self.dir

    def snell_refraction(self, normal, n1, n2):
        '''
        Refract (change direction) at a surface with a given normal (normalized) vector.
        photon.snell_refraction(self, normal, n1, n2)
        *normal*: 3-vector (normalized) of the normal
        *n1*, *n2*: indexes of refraction of inside, outside materials respectively.
        '''
        nr= n1/float(n2)
        costh1= self.dir.dot(normal)
        sin2th1= 1-costh1*costh1
        sin2th2= nr*nr*sin2th1
        rcos = msqrt( (1-sin2th2)/(1-sin2th1) ) # costh2/costh1
        self.dir= nr*self.dir + normal*(self.dir.dot(normal)) * (rcos - nr )

    def smear(self, normal, sigma):
        '''
        Randomize the direction on to the surface of a cone, whose axis is the
        original direction, the half opening angle of the cone is a Gaussian
        distribution with width sigma (in degrees). The result is confined
        withing the same side of the interface.

        smear(self, normal, sigma):
        *normal*: 3-vector (normalized) of the normal
        *sigma*: diffusion Gaussian sigma
        '''
        niter=0
        while ( True and niter<100 ):
            niter= niter+1
            # random phi
            xphi = nprnd.uniform(-np.pi, np.pi)
            # Gaussian theta. sigdif in degrees
            xtheta = nprnd.randn() * mradians(sigma)   # Random angle in radians
            xsth= msin(xtheta)
            xcth= mcos(xtheta)
            # random vector in upper hemisphere, around z-axis
            vrand = np.array([xsth*mcos(xphi), xsth*msin(xphi), xcth])
            # rotate the vector so that its z axis points to the original direction
            vrand = rotate_vector( vrand, self.dir)
            # If vrand is less than 90 degrees of normal, ok, break
            if ( vrand.dot(normal) > 1e-6 ):
                # Consistency check
                # angle between original direction and smeared direction
                if abs(mcos(xtheta)-vrand.dot(self.dir)) > 1e-6 :
                    print xtheta, vrand, self.dir
                    raise ValueError('Calculation is wrong')

                self.dir = vrand
                break

    def specular_relection(self, normal):
        '''
        Perfect specular reflection at a surface with a given normal (normalized) vector.
        diffuse_reflection(self, normal, sigma)
        *normal*: 3-vector (normalized) of the normal
        '''
        self.dir = self.dir - 2*self.dir.dot(normal) * normal


    def diffuse_reflection(self, normal, sigma):
        '''
        Diffused reflection at a surface with a given normal (normalized) vector.
        diffuse_reflection(self, normal, sigma)
        *normal*: 3-vector (normalized) of the normal
        *sigma*: diffusion Gaussian sigma
        '''
        # Perfect specular reflection first
        self.specular_relection(normal)
        # Smear
        self.smear(normal, sigma)

    def random_reflection(self, normal):
        '''
        Random reflection at a surface with a given normal (normalized) vector.
        random_reflection(self, normal)
        '''
        sign= np.sign( self.dir.dot(normal) )
        self.dir= random_direction(-sign* normal)

    def random_transmission(self, normal):
        '''
        Random transmission at a surface with a given normal (normalized) vector.
        random_transmission(self, normal)
        '''
        sign= np.sign( self.dir.dot(normal) )
        self.dir= random_direction(sign* normal)

    def transition_at_plane(self, normal, sigdif, pdif, prand, n1, n2):
        '''
        Reflected or transmitted at a plane surface
        transition_at_plane(self, sigdif, pdif, prand)
        *normal*: normal vector of the surface
        *sigdif*: diffusion sigma
        *pdif*: diffusion probability
        *prand*: random probability
        *n1*: index of refraction inside
        *n2*: index of refraction outside
               ==>> if either n1 or n2 is zero, the interface is not transparent
        '''
        self.incident_costh=  normal.dot(self.dir)

        reflected=True
        if n1>0 and n2>0:
            # Probability of reflection based on Fresnel equations
            R= reflectance(n1,n2, self.incident_costh)
            if nprnd.random_sample() > R:
                reflected=False
        else:
            pass   # Assume reflection

        if reflected:
            # Check again the surface property
            rr= nprnd.random_sample()
            if rr > pdif+prand: # transmitted anyway
                # random transmission
                self.random_transmission(normal)
                reflected= False
            elif rr > pdif: # random reflection
                self.random_reflection(normal)
                reflected= True
            else:   # diffused reflection
                self.diffuse_reflection(normal, sigdif)
                reflected= True
        else:
            # according to Snell's law
            self.snell_refraction(normal, n1, n2)
            reflected= False

        return reflected


    def reflect(self, plane, sensors):
        '''
        Change direction as it reflects at a surface and add this position to 
        the list of vertices.
        Return the plane the photon is on.
        *plane*: The plane the photon is reflecting on
        *sensors*: a list of Plane object that represents the sensors
        '''
        theplane= plane
        for sp in sensors:
            # If the photon is inside the sensor area, use the sensor
            if sp.contain( self.x ):
                theplane= sp
                break
        
        n_in= theplane.idx_refract_in
        n_out= theplane.idx_refract_out

        self.lastplane = theplane
        normal= theplane.normal
        if self.trackvtx:
            self.vertices.append( self.x )
        self.dir = unit_vect(self.dir)

        # At the crystal surface
        reflected= self.transition_at_plane(normal, theplane.sigdif_crys,
                                            theplane.pdif_crys, theplane.prand_crys,
                                            n_in, n_out)

        if reflected: 
            return theplane

        if not theplane.wrapped:  # Transmitted and no wrapper
            self.alive= False
            return theplane

        nbounces=0
        while ( not reflected and nbounces<5 ):
            nbounces= nbounces+1
            # In this loop, photon bounces between the wrapper and the crystal
            wref= self.transition_at_plane(normal,theplane.sigdif_wrap,
                                           theplane.pdif_wrap, theplane.prand_wrap,
                                           0,0)
            if not wref:  # No reflection from the wrapper
                self.alive= False
                return theplane

            # Reflected from the wrapper
            # get back in the crystal.
            # reverse normal and switch indices of reflection
            cref= self.transition_at_plane(-normal, theplane.sigdif_crys,
                                           theplane.pdif_crys, theplane.prand_crys,
                                           n_out, n_in)
            if not cref: # transmitted back into the crystal
                reflected= True

        if not reflected: self.alive= False
        return theplane


    def path_to(self, aplane):
        '''
        Pathlength (signed) to a plane.
        '''
        dom = self.dir.dot(aplane.normal)
        if abs(dom)<1e-6:
            return 1e6
        else:
            return aplane.normal.dot(aplane.corners[0] - self.x) / dom

    def nearest_plane(self, plist):
        '''
        Return the index of the next plane and the distance to that plane
        (index, distance)
        '''
        dmin = 1e9
        imin = None
        for i in xrange(len(plist)):
            #print self.x
            #print plist[i].print_properties()
            d = self.path_to(plist[i])
            if ( d < 0 ): continue
            if d < dmin:
                dmin = d
                imin = i
        return imin, dmin

    def propagate(self, crystal, verbose=False):
        '''
        Propagating this photon.
        Return the plane this photon dies on.
        '''
        rp= None
        while self.alive:
            # Check if it is inside the crystal
            if not crystal.contain(self.x):
                self.alive= False
                self.status= self.outofvolume
                if verbose:
                    print 'Outside the crystal. Kill the photon'
                    return None

            if verbose:
                print 'propagate, from point ', self.x, 'at direction', self.dir
            if not self.alive:
                if verbose: print 'Dead on arrival'
                return None
            if (abs(self.x[0])>self.mother_box[0] or
                abs(self.x[1])>self.mother_box[1] or
                abs(self.x[2])>self.mother_box[2] ):
                self.alive=False
                self.status= self.outofvolume
                if verbose: print 'Outside the mother volume', self.x
                return None

            # find the next plane
            imin,dmin= self.nearest_plane(crystal.planes)
            if ( imin == None ):  # did not find any plane
                self.alive = False
                self.status= self.noplane
                if ( verbose ): print 'Cannot find next plane'
                return None
            if ( verbose ):
                print 'path to next plane is', dmin
                crystal.planes[imin].print_properties()
            # advance photon
            self.advance(dmin, crystal.planes[imin].idx_refract_in)
            if ( not self.alive ): return None

            # reflect on this plane
            rp= self.reflect(crystal.planes[imin], crystal.sensors)
            if self.alive:
                self.n_reflects += 1
                # to stay away from this plane. 
                #  FIXME!! This will make the edges leak.
                self.advance(crystal.planes[imin].tolerance)
            else:
                self.deposit_at= self.lastplane
                self.status= self.transmitted 


        return rp

    def print_properties(self):
        '''
        Print out the property of this photon
        '''
        print 'Photon:'
        print '  start t=', self.t0, '  start x= ', self.startx
        print '  end t=', self.t, '  end x= ', self.x
        print '  status=', self.status_names[self.status]
        print '  pathlen=', self.pathlength
        print '  n_reflects=', self.n_reflects

def run_exp(crystal, zpoints, dz, dr, nperz, mfp, verbose=False):
    '''
    Run experiments. Return photon detection efficiency (and error) for each z points. 
    Photons are generated in a region specified by zpoints, dz (range in z (+-dz)) and dr (a circle with radius dr in x-y plane). 
    Directions are random in 4pi.
    Return value is an array of efficiencies and an array of errors
    *crystal*: an instance of Crystal class
    *zpoints*: an array of the z coordinates
    *nperz*: number of photons per z location
    *mfp*: mean free path of the photon
    '''
    effs, errs= [], []
    for zp in zpoints:
        if verbose:
            print zp,
        ndet=0
        for i in range(nperz):
            xp,rp= generate_p6(zp, dz, dr)
            photon= Photon(xp,rp, mfp= mfp)
            plane= photon.propagate(crystal)
            if photon.status != photon.transmitted: continue
            if photon.lastplane is None: continue
            if photon.lastplane.sensitive:
                ndet= ndet+1
            
        ef= ndet/float(nperz)
        er= np.sqrt( ef*(1-ef) / float(nperz) )
        effs.append(ef)
        errs.append(er)
    if verbose:
        print 
    return np.array(effs), np.array(errs)


def draw_one_crystal(ax, crystal, photon=None, elev=20, azim=40, xlim=(-3,3), ylim=(-3,3), zlim=(-3,3)):
    '''
    Draw one crystal in 3D.
    draw_one_crystal(ax, crystal, photon=None, elev=20, azim=40, xlim=(-3,3), ylim=(-3,3)):
    *ax*: An axis instance with projection='3d'. E.g., fig.add_subplot(111,projection='3d')
    *crystal*: An instance of Crystal class
    *photon*: An instance of Photon class to be drawn.
    '''
    ax.view_init(elev= elev, azim=azim)
    crystal.draw(ax, photon)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

