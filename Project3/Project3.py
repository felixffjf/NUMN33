from dune.grid import cartesianDomain  # type: ignore
from dune.grid import yaspGrid as yaspView # type: ignore
from dune.alugrid import aluSimplexGrid as leafGridView # type: ignore
# ----------------- Task 4 -------------------
dim = 2
domain = cartesianDomain([0]*dim, [1]*dim, [20]*dim)
gridView = yaspView(domain)
#gridView = leafGridView(domain)

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
LHS = inner(K*grad(u), grad(v))*dx
RHS = f*v*dx

from dune.fem.scheme import galerkin # type: ignore
scheme = galerkin([LHS==RHS,dbc])
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
A = 0.5*(x[0]**2 + x[1]**2) - (1/3)*(x[0]**3 - x[1]**3)
exact = exp(-2*t) * A + 1
dtExact = -2.0 * exp(-2*t) * A

def K(w):
    gradnorm = sqrt(inner(grad(w), grad(w)))
    return 2/(1 + sqrt(1 + 4*gradnorm))

f = dtExact - div( K(exact) * grad(exact) )
g_vec  = K(exact) * grad(exact)
gN = inner(g_vec, n)


#Weak form
LHS = ( (u - uh_n)/dt * v + inner(K(u)*grad(u), grad(v)) ) * dx
RHS = ( f * v ) * dx + ( gN * v ) * ds

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
#err.plot()

# ----------------- Task 7 -------------------

dt = Constant(0.01, name="dt") # time step
t = Constant(0, name="t") # current time
uh_n = uh.copy("previous") #u^n, uh = u^n+1
u, v = TrialFunction(space), TestFunction(space)
n = FacetNormal(space)


x = SpatialCoordinate(space)

exact_n = exp(-2*t) * A + 1
exact_np1 = exp(-2*(t+dt)) * A + 1
dtExact_n = -2.0 * exp(-2*t) * A
dtExact_np1 = -2.0 * exp(-2*(t+dt)) * A

f_n = dtExact_n - div( K(exact_n) * grad(exact_n) )
g_vec_n  = K(exact_n) * grad(exact_n)
gN_n = inner(g_vec_n, n)

f_np1 = dtExact_np1 - div( K(exact_np1) * grad(exact_np1) )
g_vec_np1  = K(exact_np1) * grad(exact_np1)
gN_np1 = inner(g_vec_np1, n)

#Weak form
LHS = ( (u - uh_n)/dt * v + 0.5*inner((K(u)*grad(u) + K(uh_n)*grad(uh_n)), grad(v)))*dx
RHS = 0.5*(f_n + f_np1)*v * dx + 0.5*(gN_np1 + gN_n) * v * ds

#Using Galerkin again

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


# ----------------- Task 8 -------------------
domain = cartesianDomain([0,0], [1,1], [50,50])
gridView = leafGridView(domain)
# space for solution
space = lagrange(gridView, order=1, dimRange=2)

# discrete functions needed for form
u_prev = space.interpolate([0]*2, name="u_h_prev")
u_h = space.interpolate([0]*2, name="u_h")

u = TrialFunction(space)
v = TestFunction(space)

tau = Constant(1., name="tau") # timestep constant
eps = 0.05
eps2 = Constant(eps**2, name="eps2") # we need eps*eps

def phieyre(v, w):
    return v**3 - w 

LHS_1 = (((u[0] - u_prev[0]) / tau) * v[0] + inner(grad(u[1]), grad(v[0]))) * dx
LHS_2 = (u[1] * v[1] - phieyre(u[0], u_prev[0]) * v[1] - eps2 * inner(grad(u[0]), grad(v[1]))) * dx

wf = LHS_1 + LHS_2
# ----------------- Task 9 -------------------

from ufl import conditional, sin, pi
def initial(x):
    h = 0.01
    g0 = lambda x,x0,T: conditional(x-x0<-T/2,0,conditional(x-x0>T/2,0,sin(2*pi/T*(x-x0))**3))
    G = lambda x,y,x0,y0,T: g0(x,x0,T)*g0(y,y0,T)
    return 0.5*G(x[0],x[1],0.5,0.5,50*h)

x = SpatialCoordinate(space)
initial_expr = initial(x)

u_init = space.interpolate([initial_expr, 0], name="solution")
u_h.assign(u_init)
u_prev.assign(u_h)


scheme = galerkin(wf == 0)


tau.value = 1e-3
T = 2
time = 0.0

while time < T:
    scheme.solve(target=u_h)
    time += tau.value
    u_prev.assign(u_h)

u_h[0].plot()
