from dune.grid import cartesianDomain  # type: ignore
from dune.grid import yaspGrid as yaspView # type: ignore
from dune.alugrid import aluSimplexGrid as leafGridView # type: ignore
# ----------------- Task 4 -------------------
dim = 2
domain = cartesianDomain([0]*dim, [1]*dim, [20]*dim)
#view = yaspView(domain)
#view.plot()

gridView = leafGridView(domain)
from dune.fem.space import lagrange # type: ignore

space = lagrange(gridView)
uh = space.function(name='uh')

from dune.ufl import Constant, DirichletBC # type: ignore
eps = Constant(1e-5)
p = Constant(2.0)
f = Constant(1.0)
g = Constant(0.0)

from ufl import inner, grad, div, dx, TrialFunction, TestFunction # type: ignore

dbc = DirichletBC(space, g)
u, v = TrialFunction(space), TestFunction(space)

K = (eps**2 + inner(grad(u), grad(u)))**((p-2)/2)
F = inner(K*grad(u), grad(v))*dx - f*v*dx
equation = F == 0

from dune.fem.scheme import galerkin # type: ignore
scheme = galerkin([equation,dbc])
scheme.solve(target=uh)
uh.plot()


# ----------------- Task 5 -------------------

from ufl import sqrt, FacetNormal, ds, SpatialCoordinate, exp, diff # type: ignore

dt = Constant(0.01, name="dt") # time step
t = Constant(0, name="t") # current time
uh_n = uh.copy("previous") #u^n, uh = u^n+1
u, v = TrialFunction(space), TestFunction(space)
n = FacetNormal(space)


x = SpatialCoordinate(space)
exact = exp(-2*t)*(0.5*(x[0]**2 + x[1]**2) - (1/3)*(x[0]**3 - x[1]**3)) + 1
dtExact = -2.0 * exp(-2*t)*(0.5*(x[0]**2 + x[1]**2) - (1/3)*(x[0]**3 - x[1]**3))


def K(w):
    gradnorm = sqrt(inner(grad(w), grad(w)))
    return 2/(1 + sqrt(1 + 4*gradnorm))

f_expr = dtExact - div( K(exact) * grad(exact) )
g_vec  = K(exact) * grad(exact)
gN_expr = inner(g_vec, n)


#Weak form
LHS = ( (u - uh_n)/dt * v + inner(K(u)*grad(u), grad(v)) ) * dx
RHS = ( f_expr * v ) * dx + ( gN_expr * v ) * ds

# ----------------- Task 6 -------------------
scheme = galerkin(LHS == RHS)
endTime = 0.25
dt.value = endTime/1000

# initial condition u^0 = u_ex(x,0)
t.value = 0.0
uh.interpolate(exact)
uh_n.assign(uh)

while t.value <= endTime:
    scheme.solve(target=uh)
    t.value += dt.value
    uh_n.assign(uh)

uh.plot()

# after the time loop
t.value = endTime

err = space.interpolate(uh - exact, name="error")  
err.plot()

