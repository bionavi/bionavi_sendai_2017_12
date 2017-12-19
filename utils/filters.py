import numpy

class SIRPF(object):
    """
    2d particle filter with resampling using wheel method 
    """
    def __init__(self, n_particles,measurement_noise, transition_noise\
                 , initial_state=None, initial_velocity=None, space_boundaries=None, dt=1.):
        self._Q = transition_noise
        self._R = measurement_noise
        self._n_particles = n_particles
        self._h = dt
        
        self._weights = numpy.ones((self._n_particles,1)) / self._n_particles
        if initial_state is not None:
            self._measurement_memory = initial_state
            self._particles = numpy.dot(numpy.random.rand(n_particles, 2), numpy.diag(self._R)) + initial_state
        elif space_boundaries is not None:
            self._measurement_memory = None
            self._boundaries = space_boundaries
            self.__sample()
        else:
            raise ValueError("No information about state space is provided!")
            
        if initial_velocity is None:
            self._velocity = numpy.zeros((1, 2))
        else:
            self._velocity = initial_velocity
        

    def __sample(self):
        self._particles = numpy.hstack((numpy.random.uniform(self._boundaries[0, 0]\
                                                             , self._boundaries[1, 0]\
                                                             , (self._n_particles, 1))\
                                        , numpy.random.uniform(self._boundaries[0, 1]\
                                                               , self._boundaries[1, 1]\
                                                               , (self._n_particles, 1))))

    def __resample(self):
        resampled_indices = []
        weights_cumsum = numpy.vstack(([0.], self._weights.cumsum(axis=1)))
        weights_cumsum[-1] = 1.
        offset = numpy.random.uniform()
        i = 0
        for j in numpy.arange(offset, offset + self._n_particles, 1) / self._n_particles:
            while weights_cumsum[i, 0] < j:
                i += 1 
            resampled_indices.append(i - 1)

        self._particles = self._particles[resampled_indices,:]
        self._weights = numpy.ones((self._n_particles,1)) / self._n_particles


    def __normal_pdf(self, x, μ=0., σ=1.):
        σ += 1e-50
        return numpy.exp(-0.5 * (x - μ)**2 / σ**2) / (numpy.sqrt(2 * numpy.pi) *  σ)


    def predict(self, velocity=None, dt=None):
        if velocity is not None:
            self._velocity = velocity
        if dt is not None:
            self._h = dt

        self._particles = self._particles + self._velocity * self._h + numpy.dot(numpy.random.randn(self._n_particles, 2), numpy.diag(self._Q))

    def update(self, measurement):
        if self._measurement_memory is not None:
            self._velocity = (measurement - self._measurement_memory) / self._h
        self._measurement_memory = measurement
        self._weights = self.__normal_pdf(measurement[0, 0], self._particles[:,[0]], self._R[0])\
        * self.__normal_pdf(measurement[0, 1], self._particles[:,[1]], self._R[1])
        self._weights += 1e-50
        self._weights /= self._weights.sum()
        
        # Checking for fitness of particles
        if 1. / (self._weights**2).sum() < self._n_particles / 2.:
            self.__resample()

    def estimate(self):
        x_h = (self._particles * self._weights).sum(axis=0)
        x_u = ((self._particles - x_h)**2 * self._weights).sum(axis=0)
        return x_h, x_u

    def particles(self):
        return self._particles.copy()
