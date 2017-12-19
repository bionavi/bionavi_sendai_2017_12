import numpy
import pandas
import dask.dataframe


def lat_lon_to_nvector(lat, lon):
    if (type(lat) == dask.dataframe.core.Series)\
     and (type(lon) == dask.dataframe.core.Series):
        x = dask.array.cos(dask.array.deg2rad(lat)) * dask.array.cos(dask.array.deg2rad(lon))
        y = dask.array.cos(dask.array.deg2rad(lat)) * dask.array.sin(dask.array.deg2rad(lon))
        z = dask.array.sin(dask.array.deg2rad(lat))
        x_y_z = dask.dataframe.concat([x, y, z], axis=1)
        x_y_z.columns = ['x', 'y', 'z']
        return x_y_z
    elif (type(lat) == pandas.core.series.Series)\
     and (type(lon) == pandas.core.series.Series):
        x = numpy.cos(numpy.deg2rad(lat)) * numpy.cos(numpy.deg2rad(lon))
        y = numpy.cos(numpy.deg2rad(lat)) * numpy.sin(numpy.deg2rad(lon))
        z = numpy.sin(numpy.deg2rad(lat))
        x_y_z = pandas.concat([x, y, z], axis=1)
        x_y_z.columns = ['x', 'y', 'z']
        return x_y_z
    else:
        x = numpy.cos(numpy.deg2rad(lat)) * numpy.cos(numpy.deg2rad(lon))
        y = numpy.cos(numpy.deg2rad(lat)) * numpy.sin(numpy.deg2rad(lon))
        z = numpy.sin(numpy.deg2rad(lat))
        x_y_z = numpy.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), axis=1)
        return x_y_z

def nvector_to_lat_lon(x, y, z):
    if (type(x) == dask.dataframe.core.Series)\
     and (type(y) == dask.dataframe.core.Series)\
     and (type(y) == dask.dataframe.core.Series):
        lat = dask.array.rad2deg(dask.array.arctan2(z, dask.array.sqrt(x * x + y * y)))
        lon = dask.array.rad2deg(dask.array.arctan2(y, x))
        lat_lon = dask.dataframe.concat([lat, lon], axis=1)
        lat_lon.columns = ['lat', 'lon']
        return lat_lon
    elif (type(x) == pandas.core.series.Series)\
     and (type(y) == pandas.core.series.Series)\
     and (type(y) == pandas.core.series.Series):
        lat = numpy.rad2deg(numpy.arctan2(z, numpy.sqrt(x * x + y * y)))
        lon = numpy.rad2deg(numpy.arctan2(y, x))
        lat_lon = pandas.concat([lat, lon], axis=1)
        lat_lon.columns = ['lat', 'lon']        
        return lat_lon
    else:
        lat = numpy.rad2deg(numpy.arctan2(z, numpy.sqrt(x * x + y * y)))
        lon = numpy.rad2deg(numpy.arctan2(y, x))
        lat_lon = numpy.concatenate((lat.reshape(-1, 1), lon.reshape(-1, 1)), axis=1)      
        return lat_lon

def radial_cross_distance(cross_point, end_points):
    """
    arguments:
    cross_point: nvector representation of the point from which cross distance is
    measured. ndarray of shape Nx3.
    end_points: tuple of the numpy array of the nvectors of end points of the line to which
    distance is measured with shapes (Nx3, Nx3).

    return: ndarray of shape Nx1 representing cross distances.
    """
    c_v = numpy.cross(end_points[0], end_points[1])
    c_v /= (numpy.linalg.norm(c_v, axis=1, keepdims=True) + 1e-50)
    return numpy.arccos(numpy.sum(c_v * cross_point, axis=1, keepdims=True)) - numpy.pi / 2.0


def surface_distance(end_points):
    """
    arguments:
    end_points: tuple of the numpy array of the nvectors of end points. shapes (Nx3, Nx3)
    return: angular surface distance of ndarray with shape Nx1
    """
    c_v = numpy.cross(end_points[0], end_points[1])
    i_v = numpy.sum(end_points[0] * end_points[1], axis=1, keepdims=True)
    return numpy.arctan2(numpy.linalg.norm(c_v, axis=1, keepdims=True), i_v)



class DouglasPeucker(object):
    """
    Ramer-Douglas-Peucker implementation
    tol: tolerance in fraction of great circle's radius.
    """
    def __init__(self, tol):
        self.tol = tol
        self.dominant_indices = []
        self.dominant_segments = []
    
    def reduce(self, trajectory_nvectors):
        self.dominant_indices = []
        self.dominant_segments = []
        self.trajectory_nvectors = trajectory_nvectors.as_matrix() if type(trajectory_nvectors) == pandas.core.frame.DataFrame else trajectory_nvectors.reshape(-1, 3)
        self.__douglas_peucker(0, trajectory_nvectors.shape[0] - 1)


    def __radial_cross_distance(self, cross_point, end_points):
        c_v = numpy.cross(end_points[0], end_points[1])
        c_v /= numpy.linalg.norm(c_v, axis=1, keepdims=True)
        return (numpy.arccos(numpy.sum(c_v * cross_point, axis=1, keepdims=True)) - numpy.pi / 2.0)

    def __douglas_peucker(self, start_index, end_index):
        n_vectors_end_points_ = (numpy.tile(self.trajectory_nvectors[start_index], (end_index - start_index, 1))\
        , numpy.tile(self.trajectory_nvectors[end_index], (end_index - start_index, 1)) )
        cross_dists_ =  numpy.abs(self.__radial_cross_distance(self.trajectory_nvectors[start_index:end_index], n_vectors_end_points_))
        
        max_dist = cross_dists_.max()
        max_dist_index = cross_dists_.argmax()
        if max_dist > self.tol:
            self.__douglas_peucker(start_index, start_index + max_dist_index)
            self.__douglas_peucker(start_index + max_dist_index, end_index)
            self.dominant_indices.append(start_index + max_dist_index)
        else:
            self.dominant_segments.append((start_index, end_index))