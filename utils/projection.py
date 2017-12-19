import dask
import numpy
import pandas



def mercator(lat, lon, shift_scale=None):
    origin = numpy.pi * 6378137.0 if shift_scale is None else shift_scale
    if (type(lat) == dask.dataframe.core.Series)\
     and (type(lon) == dask.dataframe.core.Series):
        x = lon * origin / 180.0
        y = dask.array.log(dask.array.tan(dask.array.deg2rad(45 + lat/2.0))) * origin / numpy.pi
        x_y = dask.dataframe.concat([x, y], axis=1)
        x_y.columns = ['x', 'y']
        return x_y
    elif (type(lat) == pandas.core.series.Series)\
     and (type(lon) == pandas.core.series.Series):
        x = lon * origin / 180.0
        y = numpy.log(numpy.tan(numpy.deg2rad(45 + lat/2.0))) * origin / numpy.pi
        x_y = pandas.concat([x, y], axis=1)
        x_y.columns = ['x', 'y']
        return x_y
    else:
        x = lon * origin / 180.0
        y = numpy.log(numpy.tan(numpy.deg2rad(45 + lat/2.0))) * origin / numpy.pi
        x_y = numpy.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
        return x_y
        
