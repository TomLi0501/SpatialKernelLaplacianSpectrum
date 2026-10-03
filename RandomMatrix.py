import numpy as np


class RandomMatrix(object):
    """

    uniform random number between 0 and 1 by default

    """
    def __init__(self, shape:tuple):
        self._shape = shape
        self.random_variables = {'x1': np.random.random} # uniform distribution between 0 and 1
        self.values = np.empty(shape, dtype = float)
        self.structure = np.empty(shape, dtype = np.dtypes.StringDType())
        self.structure.fill('x1')

    def set_variable(self, name:str, pdf):
        self.random_variables[name] = pdf
        ''' As ndarray is pre-defined, errors will be raised naturally'''

    def set_matrix(self, variable_matrix):
        """
        :param variable_matrix:
        :return: None
        You must make sure that the same name is used to set variables and matrix
        """
        if variable_matrix.shape != self._shape:
            raise ValueError("Shape Mismatch when Set Matrix")
        self.structure = variable_matrix

    def eigen(self):
        return np.linalg.eig(self.values)

    def eigenvalue_spacing(self):
        values = self.eigen().eigenvalues
        values.sort()
        spacing = [abs(values[i+1] - values[i]) for i in range(len(values)-1)]
        return spacing

    def sample(self): # this takes a sample of the random matrix
        self.__generate_matrix()
        return self

    def __generate_matrix(self):
        variables_samples = {}
        for var in self.random_variables.keys():
            variables_samples[var] = self.random_variables[var]() # get variables first
        for index, variable in np.ndenumerate(self.structure):
            self.values[index] = variables_samples[variable]

    def __call__(self):
        self.__generate_matrix()
        return self.values

    def __str__(self):
        self.__generate_matrix()
        return str(self.values)

    def __array__(self):
        """
        We can convert this object into ndarray straight away
        """
        self.__generate_matrix()
        return self.values

class WignersMatrix(RandomMatrix):
    """

    1. symmetrical
    2. diagonal : N(0, 2 * s^2/(N))
        off diagonal: N(0, s^2/(N))

    """
    def __init__(self, N, sigma):
        shape = (N,N) # First, It is symmetrical
        super().__init__(shape)
        self.random_variables = {'diagonal': self.__diagonal}
        self.values = np.empty(shape, dtype = float)
        self.structure = np.empty(shape, dtype = np.dtypes.StringDType())
        for i in range(N):
            for j in range(N):
                if i == j:
                    self.structure[i][j] = 'diagonal'
        for i in range(N):
            for j in range(N):
                if i != j:
                    self.random_variables[f'x_{i}{j}'] = self.__offdiagonal
                    self.structure[i][j] = f'x_{i}{j}'
                    self.structure[j][i] = f'x_{i}{j}'

        self.sigma = sigma

    def __diagonal(self):
        return np.random.normal(0, 2 * self.sigma**2/self._shape[0])

    def __offdiagonal(self):
        return np.random.normal(0, self.sigma**2/self._shape[0])
