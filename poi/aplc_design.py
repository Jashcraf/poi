from prysm.mathops import np, fft
from prysm.propagation import focus_fixed_sampling, focus_fixed_sampling_backprop
from prysm import coordinates, geometry
from prysm.x.optym import F77LBFGSB
from scipy.optimize import minimize
import numpy as tnp
from tqdm import tqdm
from time import sleep
import ipdb

from .propagation import ft_fwd, ft_rev, convolve_2d
from .cost_functions import MeanSquaredErrorQuadratic as MeanSquaredError


class BaseAPLC:
    def __init__(self, amp, amp_dx, efl, wvl, dark_hole, dh_dx, fpm, ls,
                 basis, initial_amplitude, activation, weight,
                 cost_function, include_fpm, alpha):
        """
        Base class for optimizing Apodized Pupil Lyot Coronagraphs,
        not intended to be used directly. Instead, use

        'AmplitudeAPLC', or
        'PhaseAPLC'

        which inherit from this class

        Parameters
        ----------
        amp: ndarray
            entrance pupil amplitude mask
        amp_dx: float
            size of 'amp' samples in milimeters
        efl: float
            effective focal length of the beam focusing onto the coronagraph
            mask, milimeters
        wvl: float
            wavelength of light, microns
        dark_hole: ndarray
            Array of the same size / sampling as the focal plane
            defining the region to make dark. Alternatively, this
            can be a circular window if 'CoreThroughput' is used as
            the cost function
        fpm: ndarray
            focal plane complex amplitude mask
        ls: ndarray
            Lyot stop complex amplitude mask
        basis: list of ndarray or None, optional
            Modal basis with which to optimize the coronagraph. If None,
            defaults to 'zonal' optimization where each sample is optimized
            independently
        initial_amplitude: ndarray, optional
            Initial apodizer amplitude to supply to the solver. If None,
            defaults to an array of ones.
        activation: class, optional
            Activation function to apply from poi.activation or
            prysm.x.optym
        weight: float, optional
            Relative weight to apply to the optimization, defaults to 1
        cost_function: Class, member of poi.cost_functions, optional
            Object with .forward() and .reverse() methods for computing
            the forward cost and gradient of the final image-plane array.
            If None, defaults to MeanSquaredError.
        alpha: float, optional
            "Steepness" parameter for some cost functions
        """

        # Define initial amplitude if not present
        if initial_amplitude is None:
            aplc = np.ones(amp.shape, dtype=np.float64)
        
        # Set up activation function
        if activation is None:
            self.activation = Dummy(1)
        else:
            self.activation = activation


        self.amp = amp
        self.amp_select = self.amp > 1e-9
        self.amp_dx = amp_dx
        self.efl = efl
        self.wvl = wvl
        self.basis = basis
        self.dh = dark_hole
        self.dh_dx = dh_dx
        self.aplc = aplc
        self.zonal = True
        self.fpm = fpm
        self.ls = ls
        self.cost = []
        self.weight = weight
        self.alpha = alpha
        self.dh_target = 0
        self.include_fpm = include_fpm

        if cost_function is None:
            self.cost_function = MeanSquaredError(target=self.dh_target)
        else:
            self.cost_function = cost_function

    def set_optimization_method(self, zonal=False):
        self.zonal = zonal

    def update(self, x):
        
        # Convert lists to arrays if that's how they are supplied
        x = np.asarray(x)

        if not self.zonal:
            self.aplc = np.tensordot(self.basis, x, axes=(0,0))
        else:
            # activate
            self.aplc = np.zeros(self.amp.shape, dtype=np.float64)
            self.aplc[self.amp_select] = self.activation.forward(x)
        
        # NOTE: This function is not defined in this base class because
        # of how different the PhaseAPLC and AmplitudeAPLC behave. See
        # Their respective definitions for the forward and adjoint of
        # '_setup_coefficients()'
        self._setup_coefficients()

        # Apply entrance pupil
        b = self.amp * self.aplc

        # prop to focal plane mask
        B = focus_fixed_sampling(
            wavefunction=b,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.dh_dx,
            output_samples=self.dh.shape,
            shift=(0, 0),
            method='mdft')
        
        # Get contrast normalization (approx)
        self.contrast_norm = (np.abs(B)**2).max()

        # apply focal plane mask
        if self.include_fpm:
            C = B * self.fpm
        else:
            C = B

        # prop to lyot stop
        c = focus_fixed_sampling(
            wavefunction=C,
            input_dx=self.dh_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.amp_dx,
            output_samples=self.amp.shape,
            shift=(0, 0),
            method='mdft')

        # apply lyot stop
        d = c * self.ls

        # prop to coronagraphic image
        D = focus_fixed_sampling(
            wavefunction=d,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.dh_dx,
            output_samples=self.dh.shape,
            shift=(0, 0),
            method='mdft')

        I = np.abs(D)**2
        N = I / self.contrast_norm
        
        # Evaluate cost function
        # Multiplying is a little weird here because MSE gets 
        # kind of thrown off. But this makes it compatible
        # With core throughput optimization as well. Alternatively,
        # We could have self.dh = the core mask?
        E = self.cost_function.forward(N[self.dh])
        E *= self.weight

        def plot_x():
            import matplotlib.pyplot as plt
            plt.figure(figsize=[10, 5])
            plt.subplot(121)
            plt.imshow(self.amp.get(), cmap="grey")
            plt.colorbar()
            plt.subplot(122)
            plt.imshow(self.aplc.get(), cmap="inferno")
            plt.colorbar()
            plt.show()
        
        def plot_n():
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm
            plt.figure()
            plt.imshow(N.get(), cmap="inferno", norm=LogNorm(vmin=1e-10, vmax=1))
            plt.colorbar()
            plt.show()

        # Present for debugging, might have been the convolution
        # ipdb.set_trace()
        # print(f"Intermediate Cost = {E}")

        self.I = I
        self.N = N
        self.E = E
        self.cost.append(self.E)
        
        # the intermediate fields
        self.b = b
        self.B = B
        self.c = c
        self.C = C
        self.d = d
        self.D = D
        return

    def fwd(self, x):
        self.update(x)
        return self.E

    def rev(self, x):
        
        # Evaluate forward model
        self.update(x)
        
        # Backpropagate image intensity to image field gradient
        Nbar = np.zeros(self.dh.shape, dtype=np.float64)

        # Backpropagation of cost function might be erroneous
        Nbar[self.dh] = self.cost_function.reverse(self.N[self.dh])
        Nbar *= self.weight

        Ibar = Nbar / self.contrast_norm
        Dbar = 2 * Ibar * self.D

        # backprop from image to lyot stop
        dbar = focus_fixed_sampling_backprop(
            wavefunction=Dbar,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.dh_dx,
            output_samples=self.aplc.shape,
            shift=(0, 0),
            method='mdft')

        # backprop lyot stop application - conjugate permits complex ls
        cbar = self.ls.conj() * dbar

        # backprop from before stop to focal plane mask
        Cbar = focus_fixed_sampling_backprop(
            wavefunction=cbar,
            input_dx=self.dh_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.amp_dx,
            output_samples=self.I.shape, 
            shift=(0, 0),
            method='mdft')

        # backprop fpm application
        if self.include_fpm:
            Bbar = self.fpm.conj() * Cbar
        else:
            Bbar = Cbar

        # backprop from before fpm to pupil apodizer
        self.bbar = focus_fixed_sampling_backprop(
            wavefunction=Bbar,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.dh_dx,
            output_samples=self.aplc.shape,
            shift=(0, 0),
            method='mdft')
        
        # See note in self.update about this function definition
        # tl;dr, don't use BaseAPLC by itself
        self._setup_coefficients_backprop()
        
        # Backpropagate to modal basis if appropriate
        if not self.zonal:
            abar = np.tensordot(self.basis, aplcbar)
        
        # The intermediate gradients
        self.Ibar = Ibar
        self.Bbar = Bbar
        self.cbar = cbar
        self.Cbar = Cbar
        self.dbar = dbar
        self.Dbar = Dbar
        self.Nbar = Nbar

        if not self.zonal:
            self.abar = abar
            return self.abar

        else:
            xbar = self.aplcbar[self.amp_select]
            abar = self.activation.backprop(xbar)
            return abar

    def fg(self, x):
        g = self.rev(x)
        f = self.E
        return f, g


class BaseSPC:
    def __init__(self, amp, amp_dx, efl, wvl, dark_hole, dh_dx,
                 basis, initial_amplitude, activation, weight,
                 cost_function, include_fpm, alpha):
        """
        Base class for optimizing Shaped Pupil Coronagraphs,
        which do not have focal plane masks or Lyot stops
        not intended to be used directly. Instead, use

        'AmplitudeSPC', or
        'PhaseSPC'

        which inherit from this class

        Parameters
        ----------
        amp: ndarray
            entrance pupil amplitude mask
        amp_dx: float
            size of 'amp' samples in milimeters
        efl: float
            effective focal length of the beam focusing onto the coronagraph
            mask, milimeters
        wvl: float
            wavelength of light, microns
        dark_hole: ndarray
            Array of the same size / sampling as the focal plane
            defining the region to make dark. Alternatively, this
            can be a circular window if 'CoreThroughput' is used as
            the cost function
        basis: list of ndarray or None, optional
            Modal basis with which to optimize the coronagraph. If None,
            defaults to 'zonal' optimization where each sample is optimized
            independently
        initial_amplitude: ndarray, optional
            Initial apodizer amplitude to supply to the solver. If None,
            defaults to an array of ones.
        activation: class, optional
            Activation function to apply from poi.activation or
            prysm.x.optym
        weight: float, optional
            Relative weight to apply to the optimization, defaults to 1
        cost_function: Class, member of poi.cost_functions, optional
            Object with .forward() and .reverse() methods for computing
            the forward cost and gradient of the final image-plane array.
            If None, defaults to MeanSquaredError.
        alpha: float, optional
            "Steepness" parameter for some cost functions
        """

        # Define initial amplitude if not present
        if initial_amplitude is None:
            aplc = np.ones(amp.shape, dtype=np.float64)
        
        # Set up activation function
        if activation is None:
            self.activation = Dummy(1)
        else:
            self.activation = activation


        self.amp = amp
        self.amp_select = self.amp > 1e-9
        self.amp_dx = amp_dx
        self.efl = efl
        self.wvl = wvl
        self.basis = basis
        self.dh = dark_hole
        self.dh_dx = dh_dx
        self.aplc = aplc
        self.zonal = True
        self.cost = []
        self.weight = weight
        self.alpha = alpha
        self.dh_target = 0
        self.include_fpm = include_fpm

        if cost_function is None:
            self.cost_function = MeanSquaredError(target=self.dh_target)
        else:
            self.cost_function = cost_function

    def set_optimization_method(self, zonal=False):
        self.zonal = zonal

    def update(self, x):
        
        # Convert lists to arrays if that's how they are supplied
        x = np.asarray(x)

        if not self.zonal:
            self.aplc = np.tensordot(self.basis, x, axes=(0,0))
        else:
            # activate
            self.aplc = np.zeros(self.amp.shape, dtype=np.float64)
            self.aplc[self.amp_select] = self.activation.forward(x)
        
        # NOTE: This function is not defined in this base class because
        # of how different the PhaseAPLC and AmplitudeAPLC behave. See
        # Their respective definitions for the forward and adjoint of
        # '_setup_coefficients()'
        self._setup_coefficients()

        # Apply entrance pupil
        b = self.amp * self.aplc

        # prop to focal plane mask
        B = focus_fixed_sampling(
            wavefunction=b,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.dh_dx,
            output_samples=self.dh.shape,
            shift=(0, 0),
            method='mdft')
        
        # Get contrast normalization (approx)
        self.contrast_norm = (np.abs(B)**2).max()

        I = np.abs(B)**2
        N = I / self.contrast_norm
        
        # Evaluate cost function
        # Multiplying is a little weird here because MSE gets 
        # kind of thrown off. But this makes it compatible
        # With core throughput optimization as well. Alternatively,
        # We could have self.dh = the core mask?
        E = self.cost_function.forward(N[self.dh])
        E *= self.weight

        self.I = I
        self.N = N
        self.E = E
        self.cost.append(self.E)

        # the intermediate fields
        self.b = b
        self.B = B
        return

    def fwd(self, x):
        self.update(x)
        return self.E

    def rev(self, x):
        
        # Evaluate forward model
        self.update(x)
        
        # Backpropagate image intensity to image field gradient
        Nbar = np.zeros(self.dh.shape, dtype=np.float64)
        Nbar[self.dh] = self.cost_function.reverse(self.N[self.dh])
        Ibar = Nbar / self.contrast_norm * self.weight
        Bbar = 2 * Ibar * self.B

        # backprop from before fpm to pupil apodizer
        self.bbar = focus_fixed_sampling_backprop(
            wavefunction=Bbar,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.dh_dx,
            output_samples=self.aplc.shape,
            shift=(0, 0),
            method='mdft')
        
        # See note in self.update about this function definition
        # tl;dr, don't use BaseAPLC by itself
        self._setup_coefficients_backprop()
        
        # Backpropagate to modal basis if appropriate
        if not self.zonal:
            abar = np.tensordot(self.basis, aplcbar)
        
        # The intermediate gradients
        self.Ibar = Ibar
        self.Bbar = Bbar
        self.bbar = bbar
        self.Nbar = Nbar

        if not self.zonal:
            self.abar = abar
            return self.abar

        else:
            xbar = self.aplcbar[self.amp_select]
            abar = self.activation.backprop(xbar)
            return abar

    def fg(self, x):
        g = self.rev(x)
        f = self.E
        return f, g


class AmplitudeAPLC(BaseAPLC):

    def __init__(self, amp, amp_dx, efl, wvl, dark_hole, dh_dx, fpm, ls,
                 basis=None, initial_amplitude=None, activation=None, weight=1,
                 cost_function=None, include_fpm=True, alpha=None):


        super().__init__(amp, amp_dx, efl, wvl, dark_hole, dh_dx, fpm, ls,
                       basis, initial_amplitude, activation, weight,
                       cost_function, include_fpm, alpha)

    def _setup_coefficients(self):
        self.aplc = np.real(self.aplc)

    def _setup_coefficients_backprop(self):
        self.aplcbar = np.real(self.bbar)


class PhaseAPLC(BaseAPLC):

    def __init__(self, amp, amp_dx, efl, wvl, dark_hole, dh_dx, fpm, ls,
                 basis=None, initial_amplitude=None, activation=None, weight=1,
                 include_fpm=True):

        super().__init__(self, amp, amp_dx, efl, wvl, dark_hole, dh_dx, fpm, ls,
                       basis=basis, initial_amplitude=initial_amplitude,
                       activation=activation, weight=weight,
                       include_fpm=include_fpm)

    def _setup_coefficients(self):
        self.W = (2 * np.pi / self.wvl) * self.aplc
        self.aplc = np.exp(1j * self.W)

    def _setup_coefficients_backprop(self):
        self.aplcbar = np.imag(self.bbar * np.conj(self.b))
        self.aplcbar *= 2 * np.pi / self.wvl 


class AmplitudeSPC(BaseSPC):

    def __init__(self, amp, amp_dx, efl, wvl, dark_hole, dh_dx,
                 basis=None, initial_amplitude=None, activation=None, weight=1,
                 cost_function=None, include_fpm=True, alpha=None):


        super().__init__(amp, amp_dx, efl, wvl, dark_hole, dh_dx,
                       basis, initial_amplitude, activation, weight,
                       cost_function, include_fpm, alpha)

    def _setup_coefficients(self):
        self.aplc = np.real(self.aplc)

    def _setup_coefficients_backprop(self):
        self.aplcbar = np.real(self.bbar)


class PhaseSPC(BaseSPC):

    def __init__(self, amp, amp_dx, efl, wvl, dark_hole, dh_dx,
                 basis=None, initial_amplitude=None, activation=None, weight=1,
                 include_fpm=True):

        super().__init__(self, amp, amp_dx, efl, wvl, dark_hole, dh_dx,
                       basis=basis, initial_amplitude=initial_amplitude,
                       activation=activation, weight=weight,
                       include_fpm=include_fpm)

    def _setup_coefficients(self):
        self.W = (2 * np.pi / self.wvl) * self.aplc
        self.aplc = np.exp(1j * self.W)

    def _setup_coefficients_backprop(self):
        self.aplcbar = np.imag(self.bbar * np.conj(self.b))
        self.aplcbar *= 2 * np.pi / self.wvl 


class Sigmoid:
    def __init__(self, a=1, x0=0, y0=0):
        """Activation function Sigmoid(x)

        Parameters
        ----------
        a : float, optional
            scale for the activation slope, by default 1
        x0 : float, optional
            x-offset of the Sigmoid(x) function, by default 0
        y0 : float, optional
            y-offset of the Sigmoid(x) function, by default 0
        """
        self.a = a
        self.x0 = x0
        self.y0 = y0

    def forward(self, x):
        x = x - self.x0
        return (1 / (1 + np.exp(-self.a * x))) + self.y0

    def backprop(self, xbar):
        sig = self.forward(xbar) - self.y0
        return self.a * sig * (1 - sig)


class Dummy:

    def __init__(self, a):
        """Activation function that does nothing
        """
        self.a = 1
        self.x0 = 0

    def forward(self,x):
        return x

    def backprop(self, xbar):
        return xbar


class BinarizationPenalty:
    def __init__(self, weight):
        self.weight = weight

    def update(self, x):
        inner = (1 - x) * x 
        self.E = self.weight * np.sum(inner)
        return 

    def fwd(self, x):
        self.update(x)
        return self.E

    def rev(self, x):
        self.update(x)
        grad = (1 - 2*x) * self.weight * self.E
        return grad

    def fg(self, x):
        g = self.rev(x)
        f = self.E
        return f, g


class AugmentedLagrangian:
    def __init__(self, objective, constraints, constraint_vals,
                 initial_multipliers, x0, penalty=10, options=None,
                 periodic_relaxation=None):
        """
        Optimization wrapper that uses the Augmented Lagrangian Method
        to iteratively solve for the optimal lagrange multipliers of 
        multiple planes to be optimized. Based on Emiel Por's 2022 SPIE
        Proceeding

        Parameters
        ----------
        objective: Optimizer
            Objective to optimize, which has an .fg() method.
        constraints: list of Optimizers
            list of optimizers, each of which have an .fg() method. These
            will have associated constraints, given by constraint_vals.
        initial_multipliers: list of float
            Initial set of lagrange multipliers, same order as val_grads
        x0: ndarray
            Initial guess for optimizer to run
        penalty: float
            The penalty for violating constraints. Think of this as the
            size of the update made when solving for lagrange multipliers
        options: dict
            Options dictionary to pass to the L-BFGS-B optimizer
        periodic_relaxation: int
            The RMS of a gaussian kernel, in samples to smooth the solution by.
            If None, periodic relaxation is ignored. This method is meant
            only for Amplitude-type coronagraphs
        """
        self.objective = objective
        self.constraints = constraints
        self.constraint_vals = constraint_vals
        self.multipliers = initial_multipliers
        self.penalty = penalty
        self.rho = penalty # this gets dynamically updated
        self.options = options

        # Init variables
        self.x0 = x0
        self.x = x0 # this gets updated for every call to step

        # init f and g
        self.refresh() 
        self.cost = []
        self.periodic_relaxation = periodic_relaxation

        # Optionally init periodic relaxation
        if self.periodic_relaxation is not None:

            # Get the amplitude shape
            shape = self.objective.amp.shape[0]
            xx = np.linspace(-shape / 2, shape / 2, self.objective.amp.shape[0])
            xx, yy = np.meshgrid(xx, xx)
            r = np.hypot(xx, yy)
            self.kernel = np.exp(-(r / periodic_relaxation) ** 2)
        
    def _setup_multipliers(self):
        """Optional method to scale mutlipliers by the gradient
        of the constraint violation, though typically they start at
        zero
        """
        h_initial = np.array([h.fg(self.x)[0] for h in self.constraints])

        # Get the gradient
        f, g = self.objective.fg(self.x)
        f_scale = np.linalg.norm(g)

        for i, h_val in enumerate(h_initial):
            h, grad_h = self.constraints[i].fg(self.x)
            h_scale = np.linalg.norm(grad_h)

            if h_scale > 0:
                self.multipliers[i] = f_scale / h_scale
            else:
                self.multipliers[i] = 0

    def refresh(self):
        self.f = 0
        self.g = 0

    def fg(self, x):
        """
        Compute value and grad for a function with constraints weighted
        by lagrange multipliers
        """

        # reset the f, g values
        self.x = x
        
        # Evaluate the objective function
        f, g = self.objective.fg(x)

        for opt, con, val in zip(self.constraints, self.constraint_vals, self.multipliers):
            
            # Evaluate function and gradient for constraint
            _f, _g = opt.fg(x)

            # Subtract off constraint to get degree of violation
            c = _f - con

            if c <= val / self.rho:
                
                # Add to objective function
                f += -1 * val * c + (self.rho / 2) * c ** 2
                
                # Add to gradient
                g += (-1 * val + self.rho * c) * _g

            elif c > val / self.rho:
                
                # Add to objective function
                f += -1 * val ** 2 / self.rho

                # Function is constant here, so nothing to add to gradient
        self.f = f
        self.g = g

        if hasattr(f, "get"):
            return self.f.get(), self.g.get()
        else:
            return self.f, self.g

    def step(self, memory=10, maxiter=10_000):
        """
        Runs an iteration of the augmented lagrangian method
        """
        
        # Optionally apply periodic relaxation to 'kick' out of
        # present solution space. Needs to happen prior to the 
        # 'inner problem' so that the solution can be optimized
        if self.periodic_relaxation is not None:
            
            # To the frequency domain jimbo
            aperture = np.copy(self.objective.amp)
            aperture[self.objective.amp_select] = self.x
            
            aperture = convolve_2d(aperture, self.kernel, normalize=True)

            self.x = aperture[self.objective.amp_select]
        
        if hasattr(self.x, "get"):
            self.x = self.x.get()
        
        # Prysm's more GPU-friendly method
        # opt = F77LBFGSB(self.fg,
        #                 self.x,
        #                 memory=memory,
        #                 upper_bounds=tnp.ones(self.x.shape),
        #                 lower_bounds=tnp.zeros(self.x.shape))
        
        # Progress bar my beloved
        pbar = tqdm(total=self.options['maxiter'], desc="Coronagraph go brr")
        
        def callback(xk):
            # Update with current function value
            pbar.set_postfix({'f(x)': self.f})
            pbar.update(1)

        callback = None

        # Define bounds for APLC
        bounds = tnp.vstack([tnp.zeros(self.x.shape), tnp.ones(self.x.shape)]).T
        
        # Trying to work with scipy
        results = minimize(self.fg, self.x, method="L-BFGS-B", jac=True,
                           bounds=bounds, callback=callback,
                           options=self.options)
        self.x = results.x
        print(results)
        pbar.close()

        print(f"Running L-BFGS-B with maxiter={maxiter} and memory={memory}")
        print("Starting values of ") 
        f, g = self.fg(self.x)
         
        print(f"f={f}")
        print(f"g={g}")
        print(f"lambda={self.multipliers[0]}") 
        print(f"rho={self.rho}") 
        
        # try:
        #     for _ in tqdm(range(maxiter)):
        #         opt.step()
        #         
        #         # Helps to break out
        #         task_str = opt.task.tobytes().decode('utf-8').strip('\x00').strip()
        #         print(task_str)
        #         if 'CONVERGENCE' in task_str or 'STOP' in task_str:
        #             break
        #     
        # except StopIteration as e:
        #     final_result = e.value

        # Update with coronagraph solution
        # self.x = self.objective.aplc[self.objective.amp_select]

        # Update the lagrange multipliers
        cost = 0 # init cost

        for i, (opt, con, val) in enumerate(zip(self.constraints, self.constraint_vals, self.multipliers)):
            
            # Evaluate function and gradient for constraint
            _f, _g = opt.fg(self.x)

            # Subtract off constraint to get degree of violation
            c = _f - con
            print(f"Constraint Violation = {c}")
            # Change in lagrange multiplier is given by how violated the constraint is
            self.multipliers[i] = max(0, val - self.rho * c)

            # Update current cost
            cost += _f
        
        # Update the penalty
        self.rho *= self.penalty
        self.cost.append(cost)



class PAPCOptimizer:
    """An apodized pupil coronagraph optimizer, pupil is phase,
    There is no focal plane mask or lyot stop

    """
    def __init__(self, amp, amp_dx, efl, wvl, basis, dark_hole, dh_dx,
                 dh_target=1e-10, initial_amplitude=None, center_wavelength=None,
                 activation=None):
        if initial_amplitude is None:
            aplc = np.zeros(amp.shape, dtype=np.float64)

        if center_wavelength is None:
                self.c_wvl = wvl

        if activation is None:
            self.activation = Dummy(1)

        else:
            self.activation = activation


        self.amp = amp
        self.amp_select = self.amp > 1e-9
        self.amp_dx = amp_dx
        self.efl = efl
        self.wvl = wvl
        self.basis = basis
        self.dh = dark_hole
        self.dh_dx = dh_dx
        self.dh_target = dh_target
        self.aplc = aplc
        self.zonal = True
        self.cost = []

    def set_optimization_method(self, zonal=False):
        self.zonal = zonal

    def update(self, x):
        x = np.array(x)
        if not self.zonal:
            self.phs = np.tensordot(self.basis, x, axes=(0,0))
        else:
            # activate
            self.phs = np.zeros(self.amp.shape, dtype=np.float64)
            self.phs[self.amp_select] = self.activation.forward(x)

        # impose constraints
        W = (2 * np.pi / self.wvl) * self.phs
        b = self.amp * np.exp(1j * W)

        # prop to focal plane mask
        B = focus_fixed_sampling(
            wavefunction=b,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.dh_dx,
            output_samples=self.dh.shape,
            shift=(0, 0),
            method='mdft')


        I = np.abs(B)**2
        E = np.sum((I[self.dh] - self.dh_target)**2)

        self.W = W
        self.I = I
        self.E = E
        self.cost.append(np.mean(I[self.dh]))

        # the fields
        self.b = b
        self.B = B

        return

    def fwd(self, x):
        self.update(x)
        return self.E

    def rev(self, x):
        self.update(x)
        Ibar = np.zeros(self.dh.shape, dtype=np.float64)
        Ibar[self.dh] = 2*(self.I[self.dh] - self.dh_target)

        Bbar = 2 * Ibar * self.B

        # backprop from image to pupil plane
        bbar = focus_fixed_sampling_backprop(
            wavefunction=Bbar,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.dh_dx,
            output_samples=self.aplc.shape,
            shift=(0, 0),
            method='mdft')

        Wbar = 2 * np.pi / self.wvl * np.imag(bbar * np.conj(self.b))

        if not self.zonal:
            abar = np.tensordot(self.basis, Wbar)

        self.Ibar = Ibar
        self.bbar = bbar
        self.Bbar = Bbar

        if not self.zonal:
            self.abar = abar
            return self.abar
        else:
            xbar = Wbar[self.amp_select]
            abar = self.activation.backprop(xbar) #* xbar
            return abar

    def fg(self, x):
        g = self.rev(x)
        f = self.E
        return f, g


class PAPLCOptimizer:
    """An apodized pupil coronagraph optimizer, pupil is phase,
    FPM and LS are= fixed

    """
    def __init__(self, amp, amp_dx, efl, wvl, basis, dark_hole, dh_dx, fpm, ls,
                 dh_target=1e-10, initial_amplitude=None, center_wavelength=None, activation=None):
        if initial_amplitude is None:
            aplc = np.zeros(amp.shape, dtype=np.float64)

        #self.val_grads = val_grads
        #self.initial_multipliers = initial_multipliers
        #self.penalty = penalty

        if center_wavelength is None:
                self.c_wvl = wvl

        if activation is None:
            self.activation = Dummy(1)

        else:
            self.activation = activation


        self.amp = amp
        self.amp_select = self.amp > 1e-9
        self.amp_dx = amp_dx
        self.efl = efl
        self.wvl = wvl
        self.basis = basis
        self.dh = dark_hole
        self.dh_dx = dh_dx
        self.dh_target = dh_target
        self.aplc = aplc
        self.zonal = True
        self.fpm = fpm
        self.ls = ls
        self.cost = []

    def set_optimization_method(self, zonal=False):
        self.zonal = zonal

    def update(self, x):
        x = np.array(x)
        if not self.zonal:
            self.phs = np.tensordot(self.basis, x, axes=(0,0))
        else:
            # activate
            self.phs = np.zeros(self.amp.shape, dtype=np.float64)
            self.phs[self.amp_select] = self.activation.forward(x)

        # impose constraints
        W = (2 * np.pi / self.wvl) * self.phs
        b = self.amp * np.exp(1j * W)

        # prop to focal plane mask
        B = focus_fixed_sampling(
            wavefunction=b,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.dh_dx,
            output_samples=self.dh.shape,
            shift=(0, 0),
            method='mdft')

        # apply focal plane mask
        C = B * self.fpm

        # prop to lyot stop
        c = focus_fixed_sampling(
            wavefunction=C,
            input_dx=self.dh_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.amp_dx,
            output_samples=self.amp.shape,
            shift=(0, 0),
            method='mdft')

        # apply lyot stop
        d = c * self.ls

        # prop to image
        D = focus_fixed_sampling(
            wavefunction=d,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.dh_dx,
            output_samples=self.dh.shape,
            shift=(0, 0),
            method='mdft')

        I = np.abs(D)**2
        E = np.sum((I[self.dh] - self.dh_target)**2)

        self.W = W
        self.I = I
        self.E = E
        self.cost.append(np.mean(I[self.dh]))

        # the fields
        self.b = b
        self.B = B
        self.c = c
        self.C = C
        self.d = d
        self.D = D

        return

    def fwd(self, x):
        self.update(x)
        return self.E

    def rev(self, x):
        self.update(x)
        Ibar = np.zeros(self.dh.shape, dtype=np.float64)
        Ibar[self.dh] = 2*(self.I[self.dh] - self.dh_target)

        Dbar = 2 * Ibar * self.D

        # backprop from image to lyot stop
        dbar = focus_fixed_sampling_backprop(
            wavefunction=Dbar,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.dh_dx,
            output_samples=self.aplc.shape,
            shift=(0, 0),
            method='mdft')

        # backprop lyot stop application
        cbar = self.ls.conj() * dbar

        # backprop from before stop to focal plane mask
        Cbar = focus_fixed_sampling_backprop(
            wavefunction=cbar,
            input_dx=self.dh_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.amp_dx,
            output_samples=self.I.shape, # this was self.amp
            shift=(0, 0),
            method='mdft')

        # backprop fpm application
        Bbar = self.fpm.conj() * Cbar

        # backprop from before fpm to pupil apodizer
        bbar = focus_fixed_sampling_backprop(
            wavefunction=Bbar,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.dh_dx,
            output_samples=self.aplc.shape,
            shift=(0, 0),
            method='mdft')

        Wbar = 2 * np.pi / self.wvl * np.imag(bbar * np.conj(self.b))

        if not self.zonal:
            abar = np.tensordot(self.basis, Wbar)

        self.Ibar = Ibar
        self.bbar = bbar
        self.Bbar = Bbar
        self.cbar = cbar
        self.Cbar = Cbar
        self.dbar = dbar
        self.Dbar = Dbar

        if not self.zonal:
            self.abar = abar
            return self.abar
        else:
            xbar = Wbar[self.amp_select]
#             xbar_neg = xbar < 0.
#             xbar_pos = xbar >= 0.
            abar = self.activation.backprop(xbar) #* xbar
            return abar

    def fg(self, x):
        g = self.rev(x)
        f = self.E
        return f, g

class APLCOptimizer:
    """An apodized pupil coronagraph optimizer, pupil is real-valued and gray-scale,
    FPM and LS are fixed

    """
    def __init__(self, amp, amp_dx, efl, wvl, basis, dark_hole, dh_dx, fpm, ls,
                 dh_target=1e-10, initial_amplitude=None, center_wavelength=None, activation=None,
                 weight=1):
        if initial_amplitude is None:
            aplc = np.zeros(amp.shape, dtype=np.float64)

        if center_wavelength is None:
            self.c_wvl = wvl

        if activation is None:
            self.activation = Dummy(1)

        else:
            self.activation = activation


        self.amp = amp
        self.amp_select = self.amp > 1e-9
        self.amp_dx = amp_dx
        self.efl = efl
        self.wvl = wvl
        self.basis = basis
        self.dh = dark_hole
        self.dh_dx = dh_dx
        self.dh_target = dh_target
        self.aplc = aplc
        self.zonal = True
        self.fpm = fpm
        self.ls = ls
        self.cost = []
        self.weight = weight

    def set_optimization_method(self, zonal=False):
        self.zonal = zonal

    def update(self, x):
        x = np.array(x)
        if not self.zonal:
            self.aplc = np.tensordot(self.basis, x, axes=(0,0))
        else:
            # activate
            self.aplc = np.zeros(self.amp.shape, dtype=np.float64)
            self.aplc[self.amp_select] = self.activation.forward(x)

        # impose constraints
        aplc = np.real(self.aplc)
        b = self.amp * aplc

        # prop to focal plane mask
        B = focus_fixed_sampling(
            wavefunction=b,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.dh_dx,
            output_samples=self.dh.shape,
            shift=(0, 0),
            method='mdft')
        
        # Get contrast normalization (approx)
        self.contrast_norm = (np.abs(B)**2).max()

        # apply focal plane mask
        C = B * self.fpm

        # prop to lyot stop
        c = focus_fixed_sampling(
            wavefunction=C,
            input_dx=self.dh_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.amp_dx,
            output_samples=self.amp.shape,
            shift=(0, 0),
            method='mdft')

        # apply lyot stop
        d = c * self.ls

        # prop to image
        D = focus_fixed_sampling(
            wavefunction=d,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.dh_dx,
            output_samples=self.dh.shape,
            shift=(0, 0),
            method='mdft')

        I = np.abs(D)**2
        N = I / self.contrast_norm
        self.alpha = 1 / np.max(N[self.dh] - self.dh_target)

        # Trying smooth maximum
        E = -log_sum_exp(N[self.dh] - self.dh_target, alpha=self.alpha)

        # Original error function is MSE:
        #E = np.sum((N[self.dh] - self.dh_target)**2) * self.weight

        self.aplc = aplc
        self.I = I
        self.N = N
        self.E = E
        self.cost.append(np.mean(N[self.dh] - self.dh_target))

        # the fields
        self.b = b
        self.B = B
        self.c = c
        self.C = C
        self.d = d
        self.D = D

        return

    def fwd(self, x):
        self.update(x)
        return self.E

    def rev(self, x):
        self.update(x)
        Nbar = np.zeros(self.dh.shape, dtype=np.float64)
        Nbar[self.dh] = -softmax(self.N[self.dh] - self.dh_target, alpha=self.alpha)
         
        # Original backprop of mean squared error
        #Nbar[self.dh] = 2*(self.N[self.dh] - self.dh_target) * self.weight
        Ibar = Nbar / self.contrast_norm
        Dbar = 2 * Ibar * self.D

        # backprop from image to lyot stop
        dbar = focus_fixed_sampling_backprop(
            wavefunction=Dbar,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.dh_dx,
            output_samples=self.aplc.shape,
            shift=(0, 0),
            method='mdft')

        # backprop lyot stop application
        cbar = self.ls.conj() * dbar

        # backprop from before stop to focal plane mask
        Cbar = focus_fixed_sampling_backprop(
            wavefunction=cbar,
            input_dx=self.dh_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.amp_dx,
            output_samples=self.I.shape, # this was self.amp
            shift=(0, 0),
            method='mdft')

        # backprop fpm application
        Bbar = self.fpm.conj() * Cbar

        # backprop from before fpm to pupil apodizer
        bbar = focus_fixed_sampling_backprop(
            wavefunction=Bbar,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.dh_dx,
            output_samples=self.aplc.shape,
            shift=(0, 0),
            method='mdft')

        aplcbar = np.real(bbar)

        if not self.zonal:
            abar = np.tensordot(self.basis, aplcbar)

        self.Ibar = Ibar
        self.bbar = bbar
        self.Bbar = Bbar
        self.cbar = cbar
        self.Cbar = Cbar
        self.dbar = dbar
        self.Dbar = Dbar
        self.Nbar = Nbar
        self.aplcbar = aplcbar

        if not self.zonal:
            self.abar = abar
            return self.abar
        else:
            xbar = self.aplcbar[self.amp_select]
#             xbar_neg = xbar < 0.
#             xbar_pos = xbar >= 0.
            abar = self.activation.backprop(xbar) #* xbar
            return abar

    def fg(self, x):
        g = self.rev(x)
        f = self.E
        return f, g


class ThroughputOptimizer:
    """An apodized pupil coronagraph optimizer for throughput, pupil is real-valued and gray-scale,
    FPM and LS are fixed

    This is substantially more simple because no propagation is actually required
    """

    def __init__(self, amp, wvl, basis, ls, initial_amplitude=None, center_wavelength=None, relative_weight=1):
        if initial_amplitude is None:
            aplc = np.zeros(amp.shape, dtype=np.float64)

        if center_wavelength is None:
            self.c_wvl = wvl

        self.amp = amp
        self.ls = ls
        # self.amp_select = self.amp * self.ls > 1e-9
        self.amp_select = self.amp > 1e-9
        self.basis = basis
        self.aplc = aplc
        self.zonal = False
        self.cost = []
        self.eta = relative_weight

    def set_optimization_method(self, zonal=False):
        self.zonal = zonal

    def update(self, x):
        x = np.array(x)
        if not self.zonal:
            self.aplc = np.tensordot(self.basis, x, axes=(0,0))

        else:
            # activate
            self.aplc[self.amp_select] = x

        # impose constraints
        aplc = np.real(self.aplc)
        b = self.amp * aplc
        
        # Ignoring the lyot stop
        # c = self.ls[self.amp_select] * b[self.amp_select]
        c = b[self.amp_select]

        #I = np.abs(c)**2
        I = c / self.amp[self.amp_select] # make throughput sampling-independent

        # Iinv = I**-1
        # E = np.sum(I)
        E = -1 * np.sum(I)

        self.aplc = aplc
        self.I = I
        # self.Iinv = Iinv
        self.E = E * self.eta
        self.cost.append(np.mean(I))

        # save the fields
        self.b = b
        self.c = c

        return

    def fwd(self, x):
        self.update(x)
        return self.E

    def rev(self, x):
        self.update(x)
        # Iinvbar = self.Iinv * self.eta
        # Ibar = -1 * ((self.Iinv.conj()) ** -2) * Iinvbar
        Ibar = -1 * self.I * self.eta
        cbar = Ibar / self.amp[self.amp_select]
        #cbar = 2 * Ibar * self.c

        # backprop lyot stop application
        # bbar = self.ls.conj()[self.amp_select] * cbar
        bbar = cbar #* self.ls[self.amp_select]
        aplcbar = np.real(bbar)

        if not self.zonal:
            abar = np.tensordot(self.basis, aplcbar)

        self.Ibar = Ibar
        self.bbar = bbar
        self.cbar = cbar
        self.aplcbar = aplcbar

        if not self.zonal:
            self.abar = abar
            return self.abar

        else:
            xbar = self.aplcbar #[self.amp_select]
            return xbar

    def fg(self, x):
        g = self.rev(x)
        f = self.E
        return f, g



class CoreThroughputOptimizer:
    """An apodized pupil coronagraph optimizer for core throughput, pupil is real-valued and gray-scale,
    FPM and LS are fixed

    """
    def __init__(self, amp, amp_dx, efl, wvl, basis, window, dh_dx, fpm, ls, initial_amplitude=None, center_wavelength=None, relative_weight=1):
        if initial_amplitude is None:
            aplc = np.ones(amp.shape, dtype=np.float64)

        if center_wavelength is None:
            self.c_wvl = wvl

        self.amp = amp
        self.amp_select = self.amp > 1e-9
        self.amp_dx = amp_dx
        self.efl = efl
        self.wvl = wvl
        self.basis = basis
        self.dh_dx = dh_dx
        self.aplc = aplc
        self.zonal = True
        self.fpm = fpm
        self.ls = ls
        self.window = window # window the size of the PSF core
        self.cost = []
        self.eta = relative_weight
        self.total_energy = np.sum(self.amp)

    def set_optimization_method(self, zonal=False):
        self.zonal = zonal

    def update(self, x):
        x = np.array(x)
        if not self.zonal:
            self.aplc = np.tensordot(self.basis, x, axes=(0,0))

        else:
            # activate
            self.aplc[self.amp_select] = x

        # impose constraints
        aplc = np.real(self.aplc)
        b = self.amp * aplc

        # Noticing that the parts behind the Lyot Stop have zero gradient,
        # therefore, they only control contrast. What if we maximize the
        # pre-FPM core throughput?
        c = b #self.ls * b

        # prop to focal plane mask
        C = focus_fixed_sampling(
            wavefunction=c,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.dh_dx,
            output_samples=self.window.shape,
            shift=(0, 0),
            method='mdft')

        I = np.abs(C)**2
        J = I / self.total_energy
        E = np.sum((J[self.window])**2)
        
        self.aplc = aplc
        self.I = I
        self.J = J
        self.E = -E * self.eta
        self.cost.append(np.mean(J[self.window]))

        # save the fields
        self.b = b
        self.c = c
        self.C = C

        return

    def fwd(self, x):
        self.update(x)
        return self.E

    def rev(self, x):
        self.update(x)
        Jbar = - 2 * self.window * self.J * self.eta
        Ibar = Jbar / self.total_energy
        
        #Ibar = - 2 * self.window * self.I * self.eta
        Cbar = 2 * Ibar * self.C

        # backprop from image to lyot stop
        cbar = focus_fixed_sampling_backprop(
            wavefunction=Cbar,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.dh_dx,
            output_samples=self.aplc.shape,
            shift=(0, 0),
            method='mdft')

        # backprop lyot stop application
        # Noticing that the parts behind the Lyot Stop have zero gradient,
        # therefore, they only control contrast. What if we maximize the
        # pre-FPM core throughput?
        bbar = self.amp * cbar #self.ls.conj() * cbar
        aplcbar = np.real(bbar)

        if not self.zonal:
            abar = np.tensordot(self.basis, aplcbar)

        self.Ibar = Ibar
        self.bbar = bbar
        self.cbar = cbar
        self.Cbar = Cbar
        self.aplcbar = aplcbar

        if not self.zonal:
            self.abar = abar
            return self.abar
        else:
            xbar = self.aplcbar[self.amp_select]
            return xbar

    def fg(self, x):
        g = self.rev(x)
        f = self.E
        return f, g


# make a wrapper that calls fwd/reverse
class APLCWrapper:
    def __init__(self,optlist):

        """optlist is a list of APLCOptimizer2 instances"""

        self.optlist = optlist

        # init f and g
        self.f = 0
        self.g = 0
        self.cost = []

    def refresh(self):
        self.f = 0
        self.g = 0

    def fg(self,x):

        # reset the f, g values
        self.refresh()

        # just sum them
        for opt in self.optlist:
            f, g = opt.fg(x)
            self.f += f
            self.g += g

        self.cost.append(self.f)

        return self.f, self.g
