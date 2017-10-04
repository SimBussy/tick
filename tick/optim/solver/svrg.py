# License: BSD 3 clause
import numpy as np

from tick.optim.model import ModelPoisReg
from tick.optim.proj import ProjHalfSpace
from warnings import warn
from tick.optim.model.base import Model
from tick.optim.solver.base import SolverFirstOrderSto
from tick.optim.solver.base.utils import relative_distance
from tick.optim.solver.build.solver import SVRG as _SVRG

__author__ = "Stephane Gaiffas"

# TODO: preparer methodes pour set et get attributes

variance_reduction_methods_mapper = {
    'last': _SVRG.VarianceReductionMethod_Last,
    'avg': _SVRG.VarianceReductionMethod_Average,
    'rand': _SVRG.VarianceReductionMethod_Random
}


class SVRG(SolverFirstOrderSto):
    """
    Stochastic variance reduced gradient

    Parameters
    ----------
    epoch_size : `int`
        Epoch size

    rand_type : `str`
        Type of random sampling

        * if ``'unif'`` samples are uniformly drawn among all possibilities
        * if ``'perm'`` a random permutation of all possibilities is
          generated and samples are sequentially taken from it. Once all of
          them have been taken, a new random permutation is generated

    tol : `float`, default=0
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it). By default the solver does ``max_iter``
        iterations

    max_iter : `int`
        Maximum number of iterations of the solver

    verbose : `bool`, default=True
        If `True`, we verbose things, otherwise the solver does not
        print anything (but records information in history anyway)

    print_every : `int`, default = 10
        Print history information every time the iteration number is a
        multiple of ``print_every``

    record_every : `int`, default = 1
        Information along iteration is recorded in history each time the
        iteration number of a multiple of ``record_every``

    Other Parameters
    ----------------
    variance_reduction : {'last', 'avg', 'rand'}, default='last'
        Determine what is used as phase iterate for variance reduction.

        * 'last' : the phase iterate is the last iterate of the previous epoch
        * 'avg' : the phase iterate is the average over the iterates in the past
          epoch. This is really a bad idea when using sparse datasets, a
          warning will be raised in this case
        * 'rand': the phase iterate is a random iterate of the previous epoch

    Attributes
    ----------
    model : `Solver`
        The model to solve

    prox : `Prox`
        Proximal operator to solve
    """

    _attrinfos = {
        '_proj': {}
    }

    def __init__(self, step: float = None, epoch_size: int = None,
                 rand_type: str = 'unif', tol: float = 0.,
                 max_iter: int = 100, verbose: bool = True,
                 print_every: int = 10, record_every: int = 1,
                 seed: int = -1, variance_reduction: str = 'last'):

        SolverFirstOrderSto.__init__(self, step, epoch_size, rand_type,
                                     tol, max_iter, verbose,
                                     print_every, record_every, seed=seed)
        step = self.step
        if step is None:
            step = 0.

        epoch_size = self.epoch_size
        if epoch_size is None:
            epoch_size = 0

        # Construct the wrapped C++ SGD solver
        self._solver = _SVRG(epoch_size, self.tol,
                             self._rand_type, step, self.seed)

        self.variance_reduction = variance_reduction
    
    def set_model(self, model: Model):
        """Set model in the solver
            
            Parameters
            ----------
            model : `Model`
            Sets the model in the solver. The model gives the first
            order information about the model (loss, gradient, among
            other things)
            
            Returns
            -------
            output : `Solver`
            The `Solver` with given model
            """
        # We need to check that the setted model is not sparse when the
        # variance reduction method is 'avg'
        if self.variance_reduction == 'avg' and model._model.is_sparse():
            warn("'avg' variance reduction cannot be used with sparse "
                 "datasets. Please change `variance_reduction` before "
                 "passing sparse data.", UserWarning)
        SolverFirstOrderSto.set_model(self, model)
        if isinstance(model, ModelPoisReg):
            A = model.features
            # mask = model.labels > 0
            A = A[model.labels > 0, :]
            b = 1e-8 + np.zeros(A.shape[0])
            self._set('_proj', ProjHalfSpace(max_iter=1000).fit(A, b))
        return self

    def objective(self, coeffs, loss: float=None):
        return self.model.loss(coeffs) + self.prox.value(coeffs)

    @property
    def variance_reduction(self):
        return next((k for k, v in variance_reduction_methods_mapper.items()
                     if v == self._solver.get_variance_reduction()), None)

    @variance_reduction.setter
    def variance_reduction(self, val: str):
        if val not in variance_reduction_methods_mapper:
            raise ValueError(
                'variance_reduction should be one of "{}", got "{}"'.format(
                    ', '.join(sorted(variance_reduction_methods_mapper.keys())),
                    val))
        if self.model is not None:
            if val == 'avg' and self.model._model.is_sparse():
                warn("'avg' variance reduction cannot be used "
                     "with sparse datasets", UserWarning)
        self._solver.set_variance_reduction(
            variance_reduction_methods_mapper[val])

    def _solve(self, x0: np.array = None, step: float = None):
        """
        Launch the solver

        Parameters
        ----------
        x0 : np.array, shape=(n_coeffs,)
            Starting iterate for the solver

        step : float
            Step-size or learning rate for the solver

        Returns
        -------
        output : np.array, shape=(n_coeffs,)
            Obtained minimizer
        """
        from tick.optim.solver import SDCA
        if not isinstance(self, SDCA):
            if step is not None:
                self.step = step

            step, obj, minimizer, prev_minimizer = \
                self._initialize_values(x0, step, n_empty_vectors=1)
            self._solver.set_starting_iterate(minimizer)

        else:
            # In sdca case x0 is a dual vector
            step, obj, minimizer, prev_minimizer = \
                self._initialize_values(None, step, n_empty_vectors=1)
            if x0 is not None:
                self._solver.set_starting_iterate(x0)

        if isinstance(self.model, ModelPoisReg) and x0 is not None:
            x0 = self._proj.call(x0)
            self._solver.set_starting_iterate(x0)

        # At each iteration we call self._solver.solve that does a full
        # epoch
        # print(x0)
        self._handle_history(0, obj=obj, x=x0)
        for n_iter in range(self.max_iter + 1):
            prev_minimizer[:] = minimizer
            prev_obj = obj
            # Launch one epoch using the wrapped C++ solver
            self._solver.solve()
            self._solver.get_minimizer(minimizer)
            # print(minimizer)
            # The step might be modified by the C++ solver
            # step = self._solver.get_step()
            if isinstance(self.model, ModelPoisReg):
                projected_coeffs = self._proj.call(minimizer)
                minimizer[:] = projected_coeffs
                obj = self.objective(projected_coeffs)
            else:
                obj = self.objective(minimizer)
            rel_delta = relative_distance(minimizer, prev_minimizer)
            rel_obj = abs(obj - prev_obj) / abs(prev_obj)
            converged = rel_obj < self.tol
            # If converged, we stop the loop and record the last step
            # in history
            extra_history = self.extra_history(minimizer)
            self._handle_history(n_iter, force=converged, obj=obj,
                                 x=minimizer.copy(), rel_delta=rel_delta,
                                 rel_obj=rel_obj, **extra_history)
            if converged:
                break
        self._set("solution", minimizer)
        return minimizer
