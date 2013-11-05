import numpy as np
import numpy.random as nprnd
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

def rect_prism(center, dim, random_reflect, diffuse_reflect, diffuse_sigma, idx_refract_in=1, idx_refract_out=1):
    '''
    Return a rectangular prism whose edges are parallel to the axes.
    *center* is the coordinate (array of three floats) of the center of gravity
    *dim* is half length of the three edges (array of three floats)
    '''
    cns=[]
    acenter= np.array(center, dtype=np.float)
    adim= np.array(dim, dtype=np.float)
    for i in range(-1,2,2):
        for j in range(-1,2,2):
            for k in range(-1,2,2):
                cns.append( acenter+adim*[i,j,k] )
    # Four corners of each plane
    cs1 = arr([ cns[0], cns[1], cns[3], cns[2] ])
    cs2 = arr([ cns[4], cns[5], cns[7], cns[6] ])
    cs3 = arr([ cns[0], cns[2], cns[6], cns[4] ])
    cs4 = arr([ cns[0], cns[1], cns[5], cns[4] ])
    cs5 = arr([ cns[2], cns[3], cns[7], cns[6] ])
    cs6 = arr([ cns[1], cns[3], cns[7], cns[5] ])

    ref= dict(random_reflect=random_reflect,
              diffuse_reflect=diffuse_reflect,
              diffuse_sigma=diffuse_sigma,
              idx_refract_in=idx_refract_in,
              idx_refract_out= idx_refract_out )
    plist = [Plane(cs1,**ref),Plane(cs2,**ref),Plane(cs3,**ref),Plane(cs4,**ref),Plane(cs5,**ref),Plane(cs6,**ref)]
    return plist

def hex_prism(center, length, width, random_reflect, diffuse_reflect, diffuse_sigma, idx_refract_in=1, idx_refract_out=1): 
    '''
    Return a hexagonal prism whose edges are parallel to the axes.
    *center* is the coordinate (array of three floats) of the center of gravity
    *length* : prism total length in z
    *width* : hexagon size, edge to the opposite edge
    '''
    cosphs= np.cos(np.arange(6)*np.pi/3.0)
    sinphs= np.sin(np.arange(6)*np.pi/3.0)
    cx= cosphs* width/np.sqrt(3) + center[0]
    cy= sinphs* width/np.sqrt(3) + center[1]
    czp= length/2.0 + center[2]
    czn= -length/2.0 + center[2]

    # Two hexagonal faces
    cs1= arr( zip(cx, cy, czp*np.ones(6)) )
    cs2= arr( zip(cx, cy, czn*np.ones(6)) )
    # Six rectangular faces
    ccs= []
    for i in range(6):
        j= np.mod(i+1,6)
        cx1, cx2, cy1, cy2= cx[i], cx[j], cy[i], cy[j]
        cx4= arr([cx1, cx1, cx2, cx2])
        cy4= arr([cy1, cy1, cy2, cy2])
        cz4= arr([czn, czp, czp, czn])

        ccs.append( arr( zip(cx4, cy4, cz4) ) )

    ref= dict(random_reflect=random_reflect,
              diffuse_reflect=diffuse_reflect,
              diffuse_sigma=diffuse_sigma,
              idx_refract_in=idx_refract_in,
              idx_refract_out= idx_refract_out )

    plist= [Plane(cc, **ref) for cc in [cs1, cs2]+ccs ]
    return plist

# Reflectance and transmittance
# Fresnel equations http://en.wikipedia.org/wiki/Fresnel_equations
# s-polarized and p-polarized
#n1-->n2
def reflectance_s(n1, n2, cos_theta_i):
    cti= abs(cos_theta_i)
    sin_theta_i = sqrt(1- cti**2)
    if n1/n2*sin_theta_i > 1:
        return 1
    cos_theta_t = sqrt(1-(n1/n2*sin_theta_i)**2)
    ret = (n1*cti - n2* cos_theta_t)/(n1*cti + n2* cos_theta_t)
    ret = ret**2
    return ret
def transmittance_s(n1, n2, cos_theta_i):
    return 1-reflectance_s(n1,n2,cos_theta_i)
def reflectance_p(n1, n2, cos_theta_i):
    cti= abs(cos_theta_i)
    sin_theta_i = sqrt(1- cti**2)
    if n1/n2*sin_theta_i > 1:
        return 1
    cos_theta_t = sqrt(1-(n1/n2*sin_theta_i)**2)
    ret = (n1*cos_theta_t - n2* cti)/(n1*cos_theta_t + n2* cti)
    ret = ret**2
    return ret
def transmittance_p(n1, n2, cos_theta_i):
    return 1-reflectance_p(n1,n2,cos_theta_i)

def reflectance(n1, n2, cos_theta_i):
    return 0.5*(reflectance_s(n1,n2,cos_theta_i)+reflectance_p(n1,n2,cos_theta_i))
def transmittance(n1, n2, cos_theta_i):
    return 1- reflectance(n1, n2, cos_theta_i)

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
        self.startx = x
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
        self.status='alive'
        self.rangeout= 'rangeout'
        self.absorbed= 'absorbed'
        self.transmitted= 'transmitted'
        self.outofvolume= 'outofvolume'
        self.noplane= 'noplane'
        self.lastplane= None

    def advance(self, d):
        '''
        Advance a distance d.
        '''
        prob = exp( -abs(d)/self.mfp )
        if ( nprnd.uniform() > prob ):
            self.alive = False
            self.status = self.rangeout

        if ( self.alive ):
            self.x = self.path(d)
            self.pathlength += d

    def path(self, d):
        '''
        A function that returns the coordinate on the photon path
        '''
        return self.x + d* self.dir

    def reflect(self, plane, sensors):
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

        self.lastplane = plane if insensor is None else insensor

        self.vertices.append( self.x )
        self.dir = unit_vect(self.dir)
        costh= normal.dot(self.dir)

        randomed= False
        diffused= False

        # If this is a sensor, check if it is reflected or transmitted
        if insensor is not None:
            R= reflectance(ifr_in,ifr_out, costh)
            #print R, ifr_in, ifr_out, costh
            if nprnd.random_sample() > R: # transmitted
                self.alive= False
                self.deposit_at= insensor
                self.status= self.transmitted
                #print self.status
            else:
                # reflected
                randomed= False
                diffused= True
        else:
            # regular crystal boundary
            rn= nprnd.random_sample()
            if rn >= rndrf+difrf: # not reflected
                self.alive = False
                self.status = self.absorbed
                randomed= False
                diffused= False
            elif rn >= difrf:  #  random reflection
                randomed= True
                diffused= False
            else:    # diffused reflection
                randomed= False
                diffused= True


        # How is it reflected
        if self.alive and randomed:   # random reflection
            xphi = nprnd.uniform(-np.pi, np.pi)
            xcth = nprnd.uniform(0,0.999)   # cos theta
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
                xphi = nprnd.uniform(-np.pi, np.pi)
                # Gaussian theta
                xtheta = nprnd.randn() * difsig
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
            self.advance(plane.tolerance)

        return plane if insensor is None else insensor

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
        Propagating this photon (A recursive function).
        Return the plane this photon dies on.
        '''
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
            if verbose: print 'Outside the mother volume', x
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
        self.advance(dmin)
        if ( not self.alive ): return None

        # reflect on this plane
        rp= self.reflect(crystal.planes[imin], crystal.sensors)

        if not self.alive:
            return rp

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
        #  probability of random reflection (angle uniform over (-pi/2,pi/2))
        self.random_reflect = random_reflect
        #  probability of diffuse reflection (angle in Gaussian distribution
        #   within sigma = diffuse_sigma 
        self.diffuse_reflect = diffuse_reflect
        self.diffuse_sigma = diffuse_sigma
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
        print '  sensitive= ', self.sensitive

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
    Class crystal. Defined by a list of Planes, and a list of sensors
    Ex.
       crystal_1= Crystal([p1, p2, p3, p4, p5, p6], [s1, s2])
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
        self.cog= np.array( [ p.cog for p in self.planes ] ).sum(axis=0)
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
            ax.plot(pts[:,0],pts[:,1],pts[:,2], 'g-o', ms=2, mec='g')
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

# Generate random directions and random positions
def generate_p6(center, dz, dr):
    '''
    Generate random directions and random positions
    *center* is the center of generated positions
    *dz* is the range in z (-dz, +dz)
    *dr* is the range in x-y plane (a circle with radius dr)
    '''
    z= center[2]+ nprnd.uniform(-1,1)* dz
    r= np.sqrt(nprnd.uniform(0,1)) * dr
    phi= nprnd.uniform(0, 2*np.pi)
    x= r*cos(phi)
    y= r*sin(phi)
    #
    phi= nprnd.uniform(0, 2*np.pi)
    cth= nprnd.uniform(-1,1)
    sth= sqrt(1-cth**2)
    xdir= sth*cos(phi)
    ydir= sth*sin(phi)
    return np.array([x,y,z]), np.array([xdir,ydir,cth])


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

