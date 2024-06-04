import logging

from fenics import *
import dolfin as df
import numpy as np

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh, rhs_fenics_mesh

class fenics_heat_2d_3(ptype):
    r"""
    Parameters
    ----------
    c_nvars : int, optional
        Spatial resolution, i.e., numbers of degrees of freedom in space.
    t0 : float, optional
        Starting time.
    family : str, optional
        Indicates the family of elements used to create the function space
        for the trail and test functions. The default is ``'CG'``, which are the class
        of Continuous Galerkin, a *synonym* for the Lagrange family of elements, see [2]_.
    order : int, optional
        Defines the order of the elements in the function space.

    Attributes
    ----------
    V : FunctionSpace
        Defines the function space of the trial and test functions.
    M : scalar, vector, matrix or higher rank tensor
        Denotes the expression :math:`\int_\Omega u_t v\,dx`.
    K : scalar, vector, matrix or higher rank tensor
        Denotes the expression :math:`- \nu \int_\Omega \nabla u \nabla v\,dx`.
    g : Expression
        The forcing term :math:`f` in the heat equation.
    bc : DirichletBC
        Denotes the Dirichlet boundary conditions.

    References
    ----------
    .. [1] The FEniCS Project Version 1.5. M. S. Alnaes, J. Blechta, J. Hake, A. Johansson, B. Kehlet, A. Logg,
        C. Richardson, J. Ring, M. E. Rognes, G. N. Wells. Archive of Numerical Software (2015).
    .. [2] Automated Solution of Differential Equations by the Finite Element Method. A. Logg, K.-A. Mardal, G. N.
        Wells and others. Springer (2012).
    """

    dtype_u = fenics_mesh
    dtype_f = rhs_fenics_mesh

    def __init__(self, c_nvars=8, t0=0.0, family='CG', order=2):
        """Initialization routine"""

        # set logger level for FFC and dolfin
        logging.getLogger('FFC').setLevel(logging.WARNING)
        logging.getLogger('UFL').setLevel(logging.WARNING)

        # set solver and form parameters
        parameters["form_compiler"]["optimize"] = True
        parameters["form_compiler"]["cpp_optimize"] = True
        parameters['allow_extrapolation'] = True

        # set mesh and refinement (for multilevel)
        mesh = df.RectangleMesh(Point(0.0, 0.0), Point(2.0, 1.0), c_nvars*2, c_nvars)

        # define function space for future reference
        self.V = FunctionSpace(mesh, family, order)
        tmp = Function(self.V)
        print('DoFs on this level:', len(tmp.vector()[:]))

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(fenics_heat_2d_3, self).__init__(self.V)
        self._makeAttributeAndRegister(
            'c_nvars', 't0', 'family', 'order', localVars=locals(), readOnly=True
        )
        
        
        # Stiffness term (Laplace)
        u = df.TrialFunction(self.V)
        v = df.TestFunction(self.V)
        a_K = -1.0 * df.inner(df.nabla_grad(u), df.nabla_grad(v)) * df.dx

        # Mass term
        a_M = u * v * df.dx

        self.M = df.assemble(a_M)
        self.K = df.assemble(a_K)

        # set boundary values
        self.u_D = Expression('sin(a*x[0])*sin(a*x[1])*cos(t)', a=np.pi, degree=self.order, alpha=self.alpha, beta=self.beta, t=t0) 
        
        # define the Dirichlet boundary
        def boundary(x, on_boundary):
            return on_boundary    
        self.bc = df.DirichletBC(self.V, self.u_D, boundary)
        self.bc_hom = df.DirichletBC(self.V, df.Constant(0), boundary)

        # set forcing term as expression
        self.g = df.Expression(
            'sin(a*x[0])*sin(a*x[1])*(2*a*a*cos(t)-sin(t))',
            a=np.pi, 
            t=t0,
            degree=self.order,
            )
        
    def solve_system(self, rhs, factor, u0, t):
        r"""
        Dolfin's linear solver for :math:`(M - factor \cdot A) \vec{u} = \vec{rhs}`.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system.
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver (not used here so far).
        t : float
            Current time.

        Returns
        -------
        u : dtype_u
            Solution.
        """

        b = self.apply_mass_matrix(rhs)

        u = self.dtype_u(u0)
        T = self.M - factor * self.K
        self.u_D.t=t
        self.bc.apply(T, b.values.vector())
        df.solve(T, u.values.vector(), b.values.vector())

        return u
        
    def __eval_fexpl(self, u, t):
        """
        Returns
        -------
        fexpl : dtype_u
            Explicit part of the right-hand side.
        """

        self.g.t = t
        fexpl = self.dtype_u(df.interpolate(self.g, self.V))

        return fexpl

    def __eval_fimpl(self, u, t):
        """
        Returns
        -------
        fimpl : dtype_u
            Explicit part of the right-hand side.
        """
        
        tmp = self.dtype_u(self.V)
        self.K.mult(u.values.vector(), tmp.values.vector())
        fimpl = self.__invert_mass_matrix(tmp)

        return fimpl


    def eval_f(self, u, t):
        """
        Returns
        -------
        f : dtype_f
            The right-hand side.
        """

        f = self.dtype_f(self.V)
        f.impl = self.__eval_fimpl(u, t)
        f.expl = self.__eval_fexpl(u, t)
        return f
    
    
    def apply_mass_matrix(self, u):
        r"""
        Returns
        -------
        me : dtype_u
            The product :math:`M \vec{u}`.
        """

        me = self.dtype_u(self.V)
        self.M.mult(u.values.vector(), me.values.vector())

        return me
    
    def __invert_mass_matrix(self, u):
        r"""
        Returns
        -------
        me : dtype_u
            The product :math:`M^{-1} \vec{u}`.
        """

        me = self.dtype_u(self.V)

        b = self.dtype_u(u)
        M = self.M
        self.bc_hom.apply(M, b.values.vector())

        df.solve(M, me.values.vector(), b.values.vector())
        return me


    def u_exact(self, t):
        u_e = Expression('sin(a*x[0])*sin(a*x[1])*cos(t)', a=np.pi, degree=self.order, alpha=self.alpha, beta=self.beta, t=t) 
        me = self.dtype_u(interpolate(u_e, self.V), val=self.V)

        return me