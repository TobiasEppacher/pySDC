import logging

from fenics import *
import dolfin as df
import numpy as np

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh, rhs_fenics_mesh

from enum import Enum

'''
Enum for mesh type selection
- UNIT_SQUARE: Unit square mesh (ny gives number of elements in each direction)
- RECTANGLE_2x1: Rectangle mesh on domain [0,2]x[0,1] with 2x1 aspect ratio in elements (ny gives number of elements in y direction)
'''
class MeshType(Enum):
    UNIT_SQUARE = 1,
    RECTANGLE_2x1 = 2

'''
Enum for equation selection
- TRIG: Trigonometric equation (Exact solution: sin(pi*x)*sin(pi*y)*cos(t))
- POLY: Polynomial equation (Exact solution: 1 + x^2 + 3*y^2 + 1.2*t)
'''
class Equation(Enum):
    POLY = 0,
    POLY_N = 1,
    TRIG = 2,
    MIXED = 3,
    MIXED_2 = 4,
    MIXED_3 = 5,
    TEST = 6
    


class fenics_heat_2d_custom(ptype):
    dtype_u = fenics_mesh
    dtype_f = rhs_fenics_mesh
    
    def getDofCount(self):
        return len(Function(self.V).vector()[:])
    
    def __init__(self, mesh_type=MeshType.UNIT_SQUARE, equation=Equation.POLY ,ny=8, t0=0.0, family='CG', order=4):
        # Allow for fixing the boundary conditions for the residual computation
        # Necessary if imex-1st-order-mass is used
        self.fix_bc_for_residual = True
        
        # set mesh
        if mesh_type==MeshType.UNIT_SQUARE:
            self.mesh = df.UnitSquareMesh(ny, ny)
        elif mesh_type==MeshType.RECTANGLE_2x1:
            self.mesh = df.RectangleMesh(Point(0.0, 0.0), Point(2.0, 1.0), 2*ny, ny)
        else:
            raise ValueError('Unknown mesh type')
        
        # define function space for future reference
        self.V = FunctionSpace(self.mesh, family, order)
        
        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(fenics_heat_2d_custom, self).__init__(self.V)
        self._makeAttributeAndRegister(
            'mesh_type', 'equation', 'ny', 't0', 'family', 'order', localVars=locals(), readOnly=True
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
        if equation==Equation.POLY:
            self.alpha = 3
            self.beta = 1.2
            self.u_D = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t', degree=self.order, alpha=self.alpha, beta=self.beta, t=t0)
        elif equation==Equation.POLY_N:
            self.d = 0
            self.a = self.b = 2
            self.c = 8
            self.u_D = df.Expression('d + pow(x[0], a) + pow(x[1], b) + pow(t, c)', degree=self.order, a=self.a, b=self.b, c=self.c, d=self.d, t=t0)
        elif equation==Equation.TRIG:
            self.u_D = Expression('sin(a*x[0])*sin(a*x[1])*cos(t)', a=np.pi, degree=self.order, t=t0)
        elif equation==Equation.MIXED: 
            self.alpha = 3
            self.beta = 1.2
            self.u_D = Expression('(1 + x[0]*x[0] + alpha*x[1]*x[1]) * cos(beta * t)', degree=self.order, alpha=self.alpha, beta=self.beta, t=t0)
        elif equation==Equation.MIXED_2: 
            self.beta = 1.2
            self.u_D = Expression('sin(a*x[0])*sin(a*x[1])*beta*t', degree=self.order, a=np.pi, beta=self.beta, t=t0)
        elif equation==Equation.MIXED_3:
            self.u_D = Expression('sin(a*x[0]) + x[1]*x[1] + t', a=np.pi, degree=self.order, t=t0)
        elif equation==Equation.TEST:
            self.u_D = Expression('pow(e, -a*t*t)*sin(1/sqrt(2)*a*x[0])*sin(1/sqrt(2)*a*x[1])', a=np.pi, e=np.e, degree=self.order, t=t0)
        else:
            raise ValueError('Unknown equation type')
        
        # define the Dirichlet boundary
        def boundary(x, on_boundary):
            return on_boundary    
        self.bc = df.DirichletBC(self.V, self.u_D, boundary)
        self.bc_hom = df.DirichletBC(self.V, df.Constant(0), boundary)

        # set forcing term as expression
        if equation==Equation.POLY:
            self.g = df.Expression('beta - 2 - 2 * alpha', degree=self.order, alpha=self.alpha, beta=self.beta, t=t0)
        elif equation==Equation.POLY_N:
            self.g = df.Expression('c*pow(t,(c-1)) - (a*a - a)*pow(x[0],(a-2)) - (b*b - b)*pow(x[1],(b-2))', degree=self.order, a=self.a, b=self.b, c=self.c, t=t0)
        elif equation==Equation.TRIG:
            self.g = df.Expression('sin(a*x[0])*sin(a*x[1])*(2*a*a*cos(t)-sin(t))', degree=self.order, a=np.pi, t=t0)
        elif equation==Equation.MIXED:
            self.g = df.Expression('-2*cos(beta*t)*(1+alpha)-(1+x[0]*x[0]+alpha*x[1]*x[1])*beta*sin(beta*t)', degree=self.order, alpha=self.alpha, beta=self.beta, t=t0)         
        elif equation==Equation.MIXED_2:
            self.g = df.Expression('(1 + 2*a*a)*sin(a*x[0])*sin(a*x[1])*beta*t', degree=self.order, a=np.pi, beta=self.beta, t=t0)
        elif equation==Equation.MIXED_3:
            self.g = df.Expression('a*a*sin(a*x[0]) - 1', degree=self.order, a=np.pi, t=t0)
        elif equation==Equation.TEST:
            self.g = df.Expression('0', degree=self.order, t=t0)
    def solve_system(self, rhs, factor, u0, t):
        r"""
        Dolfin's linear solver for :math:`(M - factor \cdot A) \vec{u} = \vec{rhs}`.
        """
        u = self.dtype_u(u0)
        T = self.M - factor * self.K
        b = self.dtype_u(rhs)

        self.u_D.t = t

        self.bc.apply(T, b.values.vector())
        self.bc.apply(b.values.vector())

        df.solve(T, u.values.vector(), b.values.vector())

        return u

    def eval_f(self, u, t):
        """
            The right-hand side.
        """
        f = self.dtype_f(self.V)

        self.K.mult(u.values.vector(), f.impl.values.vector())

        self.g.t = t
        f.expl = self.dtype_u(df.interpolate(self.g, self.V))
        f.expl = self.apply_mass_matrix(f.expl)

        return f
    
    def fix_residual(self, res):
        """
        Applies homogeneous Dirichlet boundary conditions to the residual

        Parameters
        ----------
        res : dtype_u
              Residual
        """
        self.bc_hom.apply(res.values.vector())
        return None
    
    def apply_mass_matrix(self, u):
        r"""
            The product :math:`M \vec{u}`.
        """

        me = self.dtype_u(self.V)
        self.M.mult(u.values.vector(), me.values.vector())

        return me


    def u_exact(self, t):
        if self.equation==Equation.POLY:
            u_exact = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t', degree=self.order, alpha=self.alpha, beta=self.beta, t=t)
        elif self.equation==Equation.POLY_N:
            u_exact = df.Expression('d + pow(x[0], a) + pow(x[1], b) + pow(t, c)', degree=self.order, a=self.a, b=self.b, c=self.c, d=self.d, t=t)
        elif self.equation==Equation.TRIG:
            u_exact = Expression('sin(a*x[0])*sin(a*x[1])*cos(t)', a=np.pi, degree=self.order, t=t)
        elif self.equation==Equation.MIXED:
            u_exact = Expression('(1 + x[0]*x[0] + alpha*x[1]*x[1]) * cos(beta * t)', degree=self.order, alpha=self.alpha, beta=self.beta, t=t)
        elif self.equation==Equation.MIXED_2:
            u_exact = Expression('sin(a*x[0])*sin(a*x[1])*beta*t', degree=self.order, a=np.pi, beta=self.beta, t=t)
        elif self.equation==Equation.MIXED_3:
            u_exact = Expression('sin(a*x[0]) + x[1]*x[1] + t', a=np.pi, degree=self.order, t=t)
        elif self.equation==Equation.TEST:
            u_exact = Expression('pow(e, -a*t*t)*sin(1/sqrt(2)*a*x[0])*sin(1/sqrt(2)*a*x[1])', a=np.pi, e=np.e, degree=self.order, t=t)
        else:
            raise ValueError('Unknown equation type')
        
        me = self.dtype_u(interpolate(u_exact, self.V), val=self.V)
        return me