import numpy
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch



class Arrow3d(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs
        
    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

class Sphere(object):
    def __init__(self, radius, segs=20):
        self._radius = radius
        self._segs = segs
        u = numpy.linspace(0, 2 * numpy.pi, self._segs)
        v = numpy.linspace(0, numpy.pi, self._segs)
        self.x = self._radius * numpy.outer(numpy.cos(u), numpy.sin(v))
        self.y = self._radius * numpy.outer(numpy.sin(u), numpy.sin(v))
        self.z = self._radius * numpy.outer(numpy.ones(u.shape[0]), numpy.cos(v))

    def get_volume(self):
        return 4. * numpy.pi * self._radius**3 /3.
    

class LocalTangentPlaneCoordinateFrame(object):
    def __init__(self, origin_vector, enu=True, scale=0.5):
        self._world_origin = numpy.zeros((3, 1))
        self.o = origin_vector.reshape(3, 1)
        self._z_vector = (self.o / numpy.linalg.norm(self.o)) if enu else (-1 * self.o / numpy.linalg.norm(self.o))
        north_pole_vector = numpy.asarray([0.0, 0.0, 1.0]).reshape(3,1)
        e_vector = numpy.cross(north_pole_vector, self.o , axis=0)
        e_vector_norm = numpy.linalg.norm(e_vector)
        self._e_vector = (e_vector / e_vector_norm) if e_vector_norm != 0 else numpy.asarray([0.0, 1.0, 0.0]).reshape(3,1)
        self._n_vector = numpy.cross(self._z_vector, self._e_vector, axis=0) if enu else numpy.cross(self._e_vector, self._z_vector, axis=0)
        self._transform_matrix = numpy.hstack(( self._e_vector, self._n_vector, self._z_vector)) if enu else  numpy.hstack((self._n_vector, self._e_vector, self._z_vector))

        bases_axes_matrix = numpy.dot(self._transform_matrix, numpy.diag(numpy.ones(3)) * scale) + self.o

        self.x = bases_axes_matrix[:, [0]]
        self.y = bases_axes_matrix[:, [1]]
        self.z = bases_axes_matrix[:, [2]]
    
    def transform(self, vectors):
        return numpy.dot(self._transform_matrix, vectors) + self.o

    def x_basis_axis(self, *args, **kwargs):
        return Arrow3d([self.o[0, 0], self.x[0, 0]], [self.o[1, 0], self.x[1, 0]], [self.o[2, 0], self.x[2, 0]], *args, **kwargs)
    
    def y_basis_axis(self, options=None, *args, **kwargs):
        return Arrow3d([self.o[0, 0], self.y[0, 0]], [self.o[1, 0], self.y[1, 0]], [self.o[2, 0], self.y[2, 0]], *args, **kwargs)

    def z_basis_axis(self, options=None, *args, **kwargs):
        return Arrow3d([self.o[0, 0], self.z[0, 0]], [self.o[1, 0], self.z[1, 0]], [self.o[2, 0], self.z[2, 0]], *args, **kwargs)
    
    def o_vector(self, options=None, *args, **kwargs):
        return Arrow3d([self._world_origin[0, 0], self.o[0, 0]], [self._world_origin[1, 0], self.o[1, 0]], [self._world_origin[2, 0], self.o[2, 0]], *args, **kwargs)


class NVector(object):
    def __init__(self, endpoint):
        self.o = numpy.zeros((3, 1))
        self.e = endpoint.reshape(3, 1)


    def nvector(self, *args, **kwargs):
        return Arrow3d([self.o[0, 0], self.e[0, 0]], [self.o[1, 0], self.e[1, 0]], [self.o[2, 0], self.e[2, 0]], *args, **kwargs)



class EarthCenteredCoordinateFrame(object):
    def __init__(self, scale=0.5):
        self.o = numpy.zeros((3, 1))

        self.x = numpy.asarray([1.0, 0.0, 0.0]).reshape(3,1) * scale
        self.y = numpy.asarray([0.0, 1.0, 0.0]).reshape(3,1) * scale
        self.z = numpy.asarray([0.0, 0.0, 1.0]).reshape(3,1) * scale
    
    def x_basis_axis(self, *args, **kwargs):
        return Arrow3d([self.o[0, 0], self.x[0, 0]], [self.o[1, 0], self.x[1, 0]], [self.o[2, 0], self.x[2, 0]], *args, **kwargs)
    
    def y_basis_axis(self, options=None, *args, **kwargs):
        return Arrow3d([self.o[0, 0], self.y[0, 0]], [self.o[1, 0], self.y[1, 0]], [self.o[2, 0], self.y[2, 0]], *args, **kwargs)

    def z_basis_axis(self, options=None, *args, **kwargs):
        return Arrow3d([self.o[0, 0], self.z[0, 0]], [self.o[1, 0], self.z[1, 0]], [self.o[2, 0], self.z[2, 0]], *args, **kwargs)
    

