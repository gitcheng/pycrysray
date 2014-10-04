=== Project pycrysray: ===

Optical photon ray tracing program for crystal studies. 

There are two main classes: Photon and Plane.

Crystal geometry is defined by a set of convex polygon surfaces (Planes).

Each Plane is defined by a list of corners. Each corner is an numpy.array
of three elements. Corners in the list must be in order (clockwise or counter-
clockwise). The angle at each corner must be less than 180 degrees (convex). 
One can define properties of each surface such as reflectivity.

Photon has its position and direction, as well as other properties, such as
mean free path.

A Photon travels inside the crystal and reflects at the surfaces, until it
is killed either at the surface (reflectivity) or after certain path length
(mean free path).

 

To build:

python setup.py install

It is installed in the usual place of your python packages. You may need 
root previlege.

Import everything in your application python script

  from pycrysray import *



Geometry
--------

A solid crystal is defined by a list of instances of Class Plane.
E.g.,

mycrystal = Crystal('name', [s1, s2, s3, s4, s5, s6, ...])

An instance of Plane is defined by a list of space 3D points that define
the corners of a polygon, and the property of the surface. The parameters
include those that define reflectivity including the effect of the wrapping,
the indices of refraction on inside material and outside material, and
whether the Plane is a sensitive object or not. See the documentation for 
class Plane for more detail.

Several convenient functions help to creat geometries:

 The following return a Plane

  rectangle(center, xlen, ylen, **kwargs)
  hexagon(center, width, **kwargs)
  
 The following return a list of Planes for a solid

  ngon_taper(plane1, plane2, **kwargs)
  rect_taper(center, length, xlen1, ylen1, xlen2, ylen2, **kwargs)
  rect_prism(center, length, xlen, ylen, **kwargs)
  hex_taper(center, length, width1, width2, **kwargs)
  hex_prism(center, length, width, **kwargs)


Photon
------

 An instance of class Photon is defined by its position, direction, time,
wavelength, and mean free path. Once a photon is created, one calls 
Photon.propagate(Crystal) to let the photon propagate in the crystal
until the photon is absorbed at a surface, transmitted out of the crystal,
or ranged out. The photon will record its status and the last plane it 
encounters.


