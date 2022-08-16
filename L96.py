"""
Definition of the Lorenz96 model.

Created on 2019-04-16-12-28
Author: Stephan Rasp, raspstephan@gmail.com

Modifications made by Redouane Lguensat
* Using original Lorenz equations
* Adding AMIP and OMIP versions
"""
import sys
def in_notebook():
    """
    Returns ``True`` if the module is running in IPython kernel,
    ``False`` if in IPython shell or other Python shell.
    """
    return 'ipykernel' in sys.modules
import numpy as np
import xarray as xr
import math

if in_notebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


class L96OneLevel(object):
    def __init__(self, K=36, J=10, h=1, F=10, c=10, b=10, dt=0.001,
                 X_init=None, noprog=False):
        self.K, self.J, self.h, self.F, self.c, self.b, self.dt = K, J, h, F, c, b, dt
        self.params = [self.F]
        self.noprog = noprog
        self.X = np.random.rand(self.K) if X_init is None else X_init.copy()
        self.Y = np.zeros((self.K, self.J))
        self._history_X = [self.X.copy()]

    def _rhs(self, X):
        """Compute the right hand side of the ODE."""
        dXdt = (
                -np.roll(X, -1) * (np.roll(X, -2) - np.roll(X, 1)) -
                X + self.F
        )
        return dXdt

    def step(self):
        """Step forward one time step with RK4."""
        k1 = self.dt * self._rhs(self.X)
        k2 = self.dt * self._rhs(self.X + k1 / 2)
        k3 = self.dt * self._rhs(self.X + k2 / 2)
        k4 = self.dt * self._rhs(self.X + k3)
        self.X += 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self._history_X.append(self.X.copy())

    def iterate(self, time):
        steps = int(time / self.dt)
        for n in tqdm(range(steps), disable=self.noprog):
            self.step()

    @property
    def state(self):
        return self.X

    def set_state(self, x):
        self.X = x

    @property
    def parameters(self):
        return np.atleast_1d(self.F)

    def erase_history(self):
        self._history_X = []

    @property
    def history(self):
        da = xr.DataArray(self._history_X, dims=['time', 'x'], name='X')
        return xr.Dataset(
            {'X': da},
            coords={'time': np.arange(len(self._history_X)) * self.dt, 'x': np.arange(self.K)}
        )

    def mean_stats(self, ax=None, fn=np.mean):
        h = self.history
        return np.concatenate([
            np.atleast_1d(fn(h.X, ax)),
            np.atleast_1d(fn((h.X**2), ax)),
        ])


class L96TwoLevel(object):
    def __init__(self, K=36, J=10, h=1, F=10, c=10, b=10, dt=0.001,
                 X_init=None, Y_init=None, noprog=False, noYhist=False, save_dt=0.1,
                 integration_type='uncoupled', parameterization=None):
        # Model parameters
        self.K, self.J, self.h, self.F, self.c, self.b, self.dt = K, J, h, F, c, b, dt
        self.noprog, self.noYhist, self.integration_type = noprog, noYhist, integration_type
        self.step_count = 0
        self.save_dt = save_dt
        self.parameterization = parameterization
        if self.parameterization is not None: self.integration_type = 'parameterization'
        self.save_steps = int(save_dt / dt)
        self.X = np.random.rand(self.K) if X_init is None else X_init.copy()
        self.Y = np.zeros(self.K * self.J) if Y_init is None else Y_init.copy()
        self._history_X = [self.X.copy()]
        self._history_Y_mean = [self.Y.reshape(self.K, self.J).mean(1).copy()]
        self._history_Y2_mean = [(self.Y.reshape(self.K, self.J)**2).mean(1).copy()]
        self._history_B = [-self.h * self.c * self.Y.reshape(self.K, self.J).mean(1)]
        if not self.noYhist:
            self._history_Y = [self.Y.copy()]

    def _rhs_X_dt(self, X, Y=None, B=None):
        """Compute the right hand side of the X-ODE."""
        if Y is None:
            dXdt = (
                    -np.roll(X, -1) * (np.roll(X, -2) - np.roll(X, 1)) -
                    X + self.F + B
            )
        else:
            dXdt = (
                    -np.roll(X, -1) * (np.roll(X, -2) - np.roll(X, 1)) -
                    X + self.F - self.h * self.c * Y.reshape(self.K, self.J).mean(1)
            )
        return self.dt * dXdt

    def _rhs_Y_dt(self, X, Y):
        """Compute the right hand side of the Y-ODE."""
        dYdt = (
                       -self.b * np.roll(Y, -1) * (np.roll(Y, -2) - np.roll(Y, 1)) -
                       Y + self.h / self.J * np.repeat(X, self.J)
               ) * self.c
        return self.dt * dYdt

    def _rhs_dt(self, X, Y):
        return self._rhs_X_dt(X, Y=Y), self._rhs_Y_dt(X, Y)

    def step(self, add_B=True, B=None):
        """Integrate one time step"""
        if self.parameterization is None:
            B = -self.h * self.c * self.Y.reshape(self.K, self.J).mean(1) if B is None else B
            if self.integration_type == 'coupled':
                k1_X, k1_Y = self._rhs_dt(self.X, self.Y)
                k2_X, k2_Y = self._rhs_dt(self.X + k1_X / 2, self.Y + k1_Y / 2)
                k3_X, k3_Y = self._rhs_dt(self.X + k2_X / 2, self.Y + k2_Y / 2)
                k4_X, k4_Y = self._rhs_dt(self.X + k3_X, self.Y + k3_Y)
            elif self.integration_type == 'uncoupled':
                k1_X = self._rhs_X_dt(self.X, B=B)
                k2_X = self._rhs_X_dt(self.X + k1_X / 2, B=B)
                k3_X = self._rhs_X_dt(self.X + k2_X / 2, B=B)
                k4_X = self._rhs_X_dt(self.X + k3_X, B=B)
                # Then update Y with unupdated X
                k1_Y = self._rhs_Y_dt(self.X, self.Y)
                k2_Y = self._rhs_Y_dt(self.X, self.Y + k1_Y / 2)
                k3_Y = self._rhs_Y_dt(self.X, self.Y + k2_Y / 2)
                k4_Y = self._rhs_Y_dt(self.X, self.Y + k3_Y)

            self.X += 1 / 6 * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)
            self.Y += 1 / 6 * (k1_Y + 2 * k2_Y + 2 * k3_Y + k4_Y)
        else:  # Parameterization case
            k1_X = self._rhs_X_dt(self.X, B=0)
            k2_X = self._rhs_X_dt(self.X + k1_X / 2, B=0)
            k3_X = self._rhs_X_dt(self.X + k2_X / 2, B=0)
            k4_X = self._rhs_X_dt(self.X + k3_X, B=0)

            B = self.parameterization(self.X) if B is None else B
            self.X += 1 / 6 * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)
            if add_B: self.X += B * self.dt

        self.step_count += 1
        if self.step_count % self.save_steps == 0:
            Y_mean = self.Y.reshape(self.K, self.J).mean(1)
            Y2_mean = (self.Y.reshape(self.K, self.J)**2).mean(1)
            self._history_X.append(self.X.copy())
            self._history_Y_mean.append(Y_mean.copy())
            self._history_Y2_mean.append(Y2_mean.copy())
            self._history_B.append(B.copy())
            if not self.noYhist:
                self._history_Y.append(self.Y.copy())

    def iterate(self, time):
        steps = int(time / self.dt)
        for n in tqdm(range(steps), disable=self.noprog):
            self.step()

    @property
    def state(self):
        return np.concatenate([self.X, self.Y])

    def set_state(self, x):
        self.X = x[:self.K]
        self.Y = x[self.K:]

    @property
    def parameters(self):
        return np.array([self.F, self.h, self.c, self.b])

    def erase_history(self):
        self._history_X = []
        self._history_Y_mean = []
        self._history_Y2_mean = []
        self._history_B = []
        if not self.noYhist:
            self._history_Y = []

    @property
    def history(self):
        dic = {}
        dic['X'] = xr.DataArray(self._history_X, dims=['time', 'x'], name='X')
        dic['B'] = xr.DataArray(self._history_B, dims=['time', 'x'], name='B')
        dic['Y_mean'] = xr.DataArray(self._history_Y_mean, dims=['time', 'x'], name='Y_mean')
        dic['Y2_mean'] = xr.DataArray(self._history_Y2_mean, dims=['time', 'x'], name='Y2_mean')
        if not self.noYhist:
            dic['X_repeat'] = xr.DataArray(np.repeat(self._history_X, self.J, 1),
                                   dims=['time', 'y'], name='X_repeat')
            dic['Y'] = xr.DataArray(self._history_Y, dims=['time', 'y'], name='Y')
        return xr.Dataset(
            dic,
            coords={'time': np.arange(len(self._history_X)) * self.save_dt, 'x': np.arange(self.K),
                    'y': np.arange(self.K * self.J)}
        )

    def mean_stats(self, ax=None, fn=np.mean, computeY=False):
        h = self.history
        if computeY:
        
            Newarray = h.Y[:,:self.J].values
            for u in range(self.J):
                for v in range(u,self.J,1):
                    Newarray = np.concatenate([Newarray, (h.Y[:,u]*h.Y[:,v]).values[:,None]], axis=1)
            print(Newarray.shape)    
            return np.atleast_1d(fn(Newarray, ax))
        else:
            return np.concatenate([
                np.atleast_1d(fn(h.X, ax)),
                np.atleast_1d(fn(h.Y_mean, ax)),
                np.atleast_1d(fn((h.X ** 2), ax)),
                np.atleast_1d(fn((h.X * h.Y_mean), ax)),
                np.atleast_1d(fn(h.Y2_mean, ax))
            ])

        
class L96TwoLevelOriginal(object):
    def __init__(self, K=36, J=10, h=1., F=10., c=10., b=10., dt=0.001,
                 X_init=None, Y_init=None, noprog=False, noYhist=False, save_dt=0.1,
                 integration_type='uncoupled', parameterization=None):
        # Model parameters
        self.K, self.J, self.h, self.F, self.c, self.b, self.dt = K, J, h, F, c, b, dt
        self.G = self.h * self.c / self.b
        self.noprog, self.noYhist, self.integration_type = noprog, noYhist, integration_type
        self.step_count = 0
        self.save_dt = save_dt
        self.parameterization = parameterization
        if self.parameterization is not None: self.integration_type = 'parameterization'
        self.save_steps = int(save_dt / dt)
        self.X = np.random.rand(self.K) if X_init is None else X_init.copy()
        self.Y = np.zeros(self.K * self.J) if Y_init is None else Y_init.copy()
        self._history_X = [self.X.copy()]
        self._history_Y_mean = [self.Y.reshape(self.K, self.J).mean(1).copy()]
        self._history_Y2_mean = [(self.Y.reshape(self.K, self.J)**2).mean(1).copy()]
        self._history_Y_sum = [self.Y.reshape(self.K, self.J).sum(1).copy()]
        self._history_B = [-self.G * self.Y.reshape(self.K, self.J).sum(1).copy()]
        if not self.noYhist:
            self._history_Y = [self.Y.copy()]

    def _rhs_X_dt(self, X, Y=None, B=None):
        """Compute the right hand side of the X-ODE."""
        if Y is None:
            dXdt = (
                    -np.roll(X, -1) * (np.roll(X, -2) - np.roll(X, 1)) -
                    X + self.F + B
            )
        else:
            dXdt = (
                    - np.roll(X, -1) * (np.roll(X, -2) - np.roll(X, 1)) - X + self.F
                    - self.G * Y.reshape(self.K, self.J).sum(1) 
            )
        return self.dt * dXdt

    def _rhs_Y_dt(self, X, Y):
        """Compute the right hand side of the Y-ODE."""
        dYdt = (
                       -self.b * self.c * np.roll(Y, -1) * (np.roll(Y, -2) - np.roll(Y, 1)) -
                       self.c * Y + self.G * np.repeat(X, self.J)
               ) 
        return self.dt * dYdt

    def _rhs_dt(self, X, Y):
        return self._rhs_X_dt(X, Y=Y), self._rhs_Y_dt(X, Y)

    def step(self, add_B=True, B=None):
        """Integrate one time step"""
        if self.parameterization is None:
            B = -self.G * self.Y.reshape(self.K, self.J).sum(1) if B is None else B
            if self.integration_type == 'coupled':
                k1_X, k1_Y = self._rhs_dt(self.X, self.Y)
                k2_X, k2_Y = self._rhs_dt(self.X + k1_X / 2, self.Y + k1_Y / 2)
                k3_X, k3_Y = self._rhs_dt(self.X + k2_X / 2, self.Y + k2_Y / 2)
                k4_X, k4_Y = self._rhs_dt(self.X + k3_X, self.Y + k3_Y)
            elif self.integration_type == 'uncoupled':
                k1_X = self._rhs_X_dt(self.X, B=B)
                k2_X = self._rhs_X_dt(self.X + k1_X / 2, B=B)
                k3_X = self._rhs_X_dt(self.X + k2_X / 2, B=B)
                k4_X = self._rhs_X_dt(self.X + k3_X, B=B)
                # Then update Y with unupdated X
                k1_Y = self._rhs_Y_dt(self.X, self.Y)
                k2_Y = self._rhs_Y_dt(self.X, self.Y + k1_Y / 2)
                k3_Y = self._rhs_Y_dt(self.X, self.Y + k2_Y / 2)
                k4_Y = self._rhs_Y_dt(self.X, self.Y + k3_Y)

            self.X += 1 / 6 * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)
            self.Y += 1 / 6 * (k1_Y + 2 * k2_Y + 2 * k3_Y + k4_Y)
        else:  # Parameterization case
            k1_X = self._rhs_X_dt(self.X, B=0)
            k2_X = self._rhs_X_dt(self.X + k1_X / 2, B=0)
            k3_X = self._rhs_X_dt(self.X + k2_X / 2, B=0)
            k4_X = self._rhs_X_dt(self.X + k3_X, B=0)

            B = self.parameterization(self.X) if B is None else B
            self.X += 1 / 6 * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)
            if add_B: self.X += B * self.dt

        self.step_count += 1
        if self.step_count % self.save_steps == 0:
            Y_mean = self.Y.reshape(self.K, self.J).mean(1)
            Y_sum = self.Y.reshape(self.K, self.J).sum(1)
            Y2_mean = (self.Y.reshape(self.K, self.J)**2).mean(1)
            self._history_X.append(self.X.copy())
            self._history_Y_mean.append(Y_mean.copy())
            self._history_Y_sum.append(Y_sum.copy())
            self._history_Y2_mean.append(Y2_mean.copy())
            self._history_B.append(B.copy())
            if not self.noYhist:
                self._history_Y.append(self.Y.copy())

    def iterate(self, time):
        steps = int(time / self.dt)
        for n in tqdm(range(steps), disable=self.noprog):
            self.step()

    @property
    def state(self):
        return np.concatenate([self.X, self.Y])

    def set_state(self, x):
        self.X = x[:self.K]
        self.Y = x[self.K:]

    @property
    def parameters(self):
        return np.array([self.F, self.h, self.c, self.b])

    def erase_history(self):
        self._history_X = []
        self._history_Y_mean = []
        self._history_Y_sum = []
        self._history_Y2_mean = []
        self._history_B = []
        if not self.noYhist:
            self._history_Y = []

    @property
    def history(self):
        dic = {}
        dic['X'] = xr.DataArray(self._history_X, dims=['time', 'x'], name='X')
        dic['B'] = xr.DataArray(self._history_B, dims=['time', 'x'], name='B')
        dic['Y_mean'] = xr.DataArray(self._history_Y_mean, dims=['time', 'x'], name='Y_mean')
        dic['Y_sum'] = xr.DataArray(self._history_Y_sum, dims=['time', 'x'], name='Y_sum')
        dic['Y2_mean'] = xr.DataArray(self._history_Y2_mean, dims=['time', 'x'], name='Y2_mean')
        if not self.noYhist:
            dic['X_repeat'] = xr.DataArray(np.repeat(self._history_X, self.J, 1),
                                   dims=['time', 'y'], name='X_repeat')
            dic['Y'] = xr.DataArray(self._history_Y, dims=['time', 'y'], name='Y')
        return xr.Dataset(
            dic,
            coords={'time': np.arange(len(self._history_X)) * self.save_dt, 'x': np.arange(self.K),
                    'y': np.arange(self.K * self.J)}
        )

    def mean_stats(self, ax=None, fn=np.mean, computeY=False):
        h = self.history
        if computeY:
            h_Y = np.array(self._history_Y)
            metrics = []
            for k in range(self.K) :
                Newarray = h_Y[:,k*self.J:(k+1)*self.J]
                for u in range(self.J):
                    for v in range(u,self.J,1):
                        Newarray = np.concatenate([Newarray, (h_Y[:,u]*h_Y[:,v])[:,None]], axis=1)
                metrics.append(np.atleast_1d(fn(Newarray, ax)))
            return np.array(metrics).flatten()
            #Newarray = h.Y[:,:self.J].values
            #for u in range(self.J):
            #    for v in range(u,self.J,1):
            #        Newarray = np.concatenate([Newarray, (h.Y[:,u]*h.Y[:,v]).values[:,None]], axis=1)
            #print(Newarray.shape)    
            #return np.atleast_1d(fn(Newarray, ax))
        else:
            return np.concatenate([
                np.atleast_1d(fn(h.X, ax)),
                np.atleast_1d(fn(h.Y_mean, ax)),
                np.atleast_1d(fn((h.X ** 2), ax)),
                np.atleast_1d(fn((h.X * h.Y_mean), ax)),
                np.atleast_1d(fn(h.Y2_mean, ax))
            ])
        
class AMIP(object):
    def __init__(self, Xdata=None, K=36, J=10, h=1., F=10., c=10., b=10., dt=0.001,
                 X_init=None, Y_init=None, noprog=False, noYhist=False, save_dt=0.1):
        
        # AMIP
        self.Xdata = Xdata
        # Model parameters
        self.K, self.J, self.h, self.F, self.c, self.b, self.dt = K, J, h, F, c, b, dt
        self.noprog, self.noYhist = noprog, noYhist
        self.step_count = 0
        self.save_dt = save_dt
        self.save_steps = int(save_dt / dt)
        self.X = np.random.rand(self.K) if X_init is None else X_init.copy()
        self.Y = np.zeros(self.K * self.J) if Y_init is None else Y_init.copy()
        self._history_X = [self.X.copy()]
        self._history_Y_mean = [self.Y.reshape(self.K, self.J).mean(1).copy()]
        self._history_Y2_mean = [(self.Y.reshape(self.K, self.J)**2).mean(1).copy()]
        if not self.noYhist:
            self._history_Y = [self.Y.copy()]
    
    def _rhs_Y_dt(self, X, Y):
        """Compute the right hand side of the Y-ODE."""
        dYdt = (
                       -self.b * self.c * np.roll(Y, -1) * (np.roll(Y, -2) - np.roll(Y, 1)) -
                       self.c * Y + (self.h * self.c / self.b) * np.repeat(X, self.J)
               ) 
        return self.dt * dYdt

    def step(self):
        
        self.X = self.Xdata[self.step_count,:].copy().values
        
        """Integrate one time step"""
        # update Y with unupdated X
        k1_Y = self._rhs_Y_dt(self.X, self.Y)
        k2_Y = self._rhs_Y_dt(self.X, self.Y + k1_Y / 2)
        k3_Y = self._rhs_Y_dt(self.X, self.Y + k2_Y / 2)
        k4_Y = self._rhs_Y_dt(self.X, self.Y + k3_Y)

        self.Y += 1 / 6 * (k1_Y + 2 * k2_Y + 2 * k3_Y + k4_Y)

        self.step_count += 1
        if self.step_count % self.save_steps == 0:
            Y_mean = self.Y.reshape(self.K, self.J).mean(1)
            Y2_mean = (self.Y.reshape(self.K, self.J)**2).mean(1)
            self._history_X.append(self.Xdata[self.step_count,:].copy().values)
            self._history_Y_mean.append(Y_mean.copy())
            self._history_Y2_mean.append(Y2_mean.copy())
            if not self.noYhist:
                self._history_Y.append(self.Y.copy())

    def iterate(self, time):
        steps = int(time / self.dt)
        for n in tqdm(range(steps), disable=self.noprog):
            self.step()

    @property
    def state(self):
        return np.concatenate([self.X, self.Y])

    def set_state(self, x):
        self.X = x[:self.K]
        self.Y = x[self.K:]

    @property
    def parameters(self):
        return np.array([self.F, self.h, self.c, self.b])

    def erase_history(self):
        self._history_X = []
        self._history_Y_mean = []
        self._history_Y2_mean = []
        if not self.noYhist:
            self._history_Y = []

    @property
    def history(self):
        dic = {}
        dic['X'] = xr.DataArray(self._history_X, dims=['time', 'x'], name='X')
        dic['Y_mean'] = xr.DataArray(self._history_Y_mean, dims=['time', 'x'], name='Y_mean')
        dic['Y2_mean'] = xr.DataArray(self._history_Y2_mean, dims=['time', 'x'], name='Y2_mean')
        if not self.noYhist:
            dic['X_repeat'] = xr.DataArray(np.repeat(self._history_X, self.J, 1),
                                   dims=['time', 'y'], name='X_repeat')
            dic['Y'] = xr.DataArray(self._history_Y, dims=['time', 'y'], name='Y')
        return xr.Dataset(
            dic,
            coords={'time': np.arange(len(self._history_X)) * self.save_dt, 'x': np.arange(self.K),
                    'y': np.arange(self.K * self.J)}
        )

    def mean_stats(self, ax=None, fn=np.mean, computeY=False):
        h = self.history
        if computeY:       
            #Newarray = h.Y[:,:self.J].values
            #for u in range(self.J):
            #    for v in range(u,self.J,1):
            #        Newarray = np.concatenate([Newarray, (h.Y[:,u]*h.Y[:,v]).values[:,None]], axis=1)
            #print(Newarray.shape)    
            #return np.atleast_1d(fn(Newarray, ax))
            h_Y = np.array(self._history_Y)
            metrics = []
            for k in range(self.K) :
                Newarray = h_Y[:,k*self.J:(k+1)*self.J]
                for u in range(self.J):
                    for v in range(u,self.J,1):
                        Newarray = np.concatenate([Newarray, (h_Y[:,u]*h_Y[:,v])[:,None]], axis=1)
                metrics.append(np.atleast_1d(fn(Newarray, ax)))
            return np.array(metrics).flatten()
        else:
            return np.concatenate([
                np.atleast_1d(fn(h.X, ax)),
                np.atleast_1d(fn(h.Y_mean, ax)),
                np.atleast_1d(fn((h.X ** 2), ax)),
                np.atleast_1d(fn((h.X * h.Y_mean), ax)),
                np.atleast_1d(fn(h.Y2_mean, ax))
            ])

        
class OMIP(object):
    def __init__(self, Ysumdata=None, K=36, J=10, F=10., G=1., dt=0.001,
                 X_init=None, Y_init=None, noprog=False, noYhist=False, save_dt=0.1):
        
        # OMIP
        self.Ysumdata = Ysumdata
        # Model parameters
        self.K, self.J, self.F, self.G, self.dt = K, J, F, G, dt
        self.noprog, self.noYhist = noprog, noYhist
        self.step_count = 0
        self.save_dt = save_dt
        self.save_steps = int(save_dt / dt)
        self.X = np.random.rand(self.K) if X_init is None else X_init.copy()
        self._history_X = [self.X.copy()]
        self._history_Y_mean = [self.Ysumdata[self.step_count,:].copy().values/self.J]

    def _rhs_X_dt(self, X, B):
        """Compute the right hand side of the X-ODE."""
        dXdt = (-np.roll(X, -1) * (np.roll(X, -2) - np.roll(X, 1)) - X + self.F + B)
        return self.dt * dXdt

    def step(self, add_B=True, B=None):
        """Integrate one time step"""
        B = -self.G * self.Ysumdata[self.step_count,:].copy().values if B is None else B
        
        k1_X = self._rhs_X_dt(self.X, B=B)
        k2_X = self._rhs_X_dt(self.X + k1_X / 2, B=B)
        k3_X = self._rhs_X_dt(self.X + k2_X / 2, B=B)
        k4_X = self._rhs_X_dt(self.X + k3_X, B=B)

        self.X += 1 / 6 * (k1_X + 2 * k2_X + 2 * k3_X + k4_X)

        self.step_count += 1
        if self.step_count % self.save_steps == 0:
            Y_mean = self.Ysumdata[self.step_count,:].copy().values/self.J
            self._history_X.append(self.X.copy())
            self._history_Y_mean.append(Y_mean.copy())

    def iterate(self, time):
        steps = int(time / self.dt)
        for n in tqdm(range(steps), disable=self.noprog):
            self.step()

    @property
    def state(self):
        return np.concatenate([self.X, self.Y])

    def set_state(self, x):
        self.X = x[:self.K]
        self.Y = x[self.K:]

    @property
    def parameters(self):
        return np.array([self.G, self.F])

    def erase_history(self):
        self._history_X = []
        self._history_Y_mean = []
        self._history_Y2_mean = []

    @property
    def history(self):
        dic = {}
        dic['X'] = xr.DataArray(self._history_X, dims=['time', 'x'], name='X')
        dic['Y_mean'] = xr.DataArray(self._history_Y_mean, dims=['time', 'x'], name='Y_mean')
        return xr.Dataset(
            dic,
            coords={'time': np.arange(len(self._history_X)) * self.save_dt, 'x': np.arange(self.K),
                    'y': np.arange(self.K * self.J)}
        )

    def mean_stats(self, ax=None, fn=np.mean):
        h = self.history
        return np.concatenate([
            np.atleast_1d(fn(h.X, ax)),
            np.atleast_1d(fn(h.Y_mean, ax)),
            np.atleast_1d(fn((h.X ** 2), ax)),
            np.atleast_1d(fn((h.X * h.Y_mean), ax))
        ])