import numpy as np # type: ignore
import scipy.sparse   # type: ignore
import dune.geometry #type: ignore
#Task 3
class LinearLagrangeSpace:
    def __init__(self,view):
        self.view = view
        self.mapper = view.mapper(lambda gt: 1 if gt.dim == 0 else 0) #Assigning one degree of freedom per vertex
        self.localDofs = 3
        self.points = np.array( [ [0,0],[1,0],[0,1] ] )
    def evaluateLocal(self, x):
        return np.array( [1-x[0]-x[1], x[0], x[1]] ) #Using barycentric coordinates
    def gradientLocal(self, x):
        dbary = [[-1,-1], [1,0], [0,1]]
        return np.array( dbary )

#Task 4
def assemble(space,force):
    # storage for right hand side
    rhs = np.zeros(len(space.mapper))

    # storage for local matrix
    localEntries = space.localDofs
    localMatrix = np.zeros([localEntries,localEntries])

    # data structure for global matrix using coordinate (COO) format
    globalEntries = localEntries**2 * space.view.size(0)
    value = np.zeros(globalEntries)
    rowIndex, colIndex = np.zeros(globalEntries,int), np.zeros(globalEntries,int)

    # TODO: implement assembly of matrix and forcing term
    dim = space.view.dimension
    localRhs = np.zeros(localEntries)
    p = 0

    for E in space.view.elements:
        geo = E.geometry
        localMatrix.fill(0.0)
        localRhs.fill(0.0)

        # quadrature 
        quad = dune.geometry.quadratureRule(E.type, 2)

        for qp in quad:
            xhat = qp.position
            wq = qp.weight

            x = geo.toGlobal(xhat)
            detJ = geo.integrationElement(xhat)

            phi = space.evaluateLocal(xhat)
            fx = force(x)

            w = wq * detJ

            # local rhs:
            for l in range(localEntries):
                localRhs[l] += w * fx * phi[l]

            # local mass matrix
            for l in range(localEntries):
                for k in range(localEntries):
                    localMatrix[l,k] += w * phi[l] * phi[k]

        # local-to-global vertex dof map g_E(l)
        g = [space.mapper.subIndex(E, l, dim) for l in range(localEntries)]

        # store rhs elements
        for l in range(localEntries):
            rhs[g[l]] += localRhs[l]

        # store local matrix into COO arrays
        for l in range(localEntries):
            for k in range(localEntries):
                rowIndex[p] = g[l]
                colIndex[p] = g[k]
                value[p] = localMatrix[l,k]
                p += 1

    # convert data structure to compressed row storage (csr)
    matrix = scipy.sparse.coo_matrix((value, (rowIndex, colIndex)),
    shape=(len(space.mapper),len(space.mapper))).tocsr()
    return rhs,matrix

from dune.grid import cartesianDomain # type: ignore
from dune.alugrid import aluConformGrid # type: ignore

# First construct the grid
domain = cartesianDomain([0, 0], [1, 1], [10, 10])
view = aluConformGrid(domain)

from dune.fem.function import gridFunction # type: ignore
from dune.fem import integrate # type: ignore
# Grid function to project
@gridFunction(view, name="u_ex", order=3)
def u(p):
    x,y = p
    return np.cos(2*np.pi*x)*np.cos(2*np.pi*y)


def grad_u(p):
    x, y = p
    return np.array([
        -2*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y),
        -2*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y)
    ])

prev_L2 = None
prev_H1 = None

# Projection on a series of globally refined grids
for ref in range(3):
    space = LinearLagrangeSpace(view)
    print("number of elements:",view.size(0),"number of dofs:",len(space.mapper))
    
    rhs,matrix = assemble(space, u)
    dofs = scipy.sparse.linalg.spsolve(matrix,rhs)
    @gridFunction(view, name="u_h", order=1)
    def uh(e,x):
        indices = space.mapper(e)
        phiVals = space.evaluateLocal(x)
        localDofs = dofs[indices]
        return np.dot(localDofs, phiVals)
    
    @gridFunction(view, name="err2", order=5) # Task 5
    def err2(e, xhat):                        
        x = e.geometry.toGlobal(xhat)
        return (u(x) - uh(e, xhat))**2
    
    @gridFunction(view, name="graderr", order=5) # Task 6
    def graderr(e, xhat):
        geo = e.geometry
        x = geo.toGlobal(xhat)
        BinvT = geo.jacobianInverseTransposed(xhat)

        gradhat = np.array([[-1., -1.],
                        [ 1.,  0.],
                        [ 0.,  1.]])

        indices = space.mapper(e)
        U = dofs[indices]

        # compute grad u_h
        grad_uh = np.zeros(2)
        for k in range(3):
            grad_k = BinvT @ gradhat[k]
            grad_uh += U[k] * grad_k

        # exact gradient
        x0, x1 = x
        grad_exact = np.array([
            -2*np.pi*np.sin(2*np.pi*x0)*np.cos(2*np.pi*x1),
            -2*np.pi*np.cos(2*np.pi*x0)*np.sin(2*np.pi*x1)
        ])

        d = grad_exact - grad_uh
        return d @ d
    
    graderr2 = integrate(graderr, view, order=5)
    
    l2err2 = integrate(err2, view, order=5)   #L2 type error in accordance with the formula in manual
    L2 = np.sqrt(l2err2)
    print("L2 error:", L2)
    
    H1 = np.sqrt(l2err2 + graderr2)
    
    print("H1 norm:", H1)
    
    if prev_L2 is not None:
        print("L2 EOC:", np.log(prev_L2 / L2) / np.log(2))
        print("H1 EOC:", np.log(prev_H1 / H1) / np.log(2))
    
    prev_L2 = L2
    prev_H1 = H1


    
    uh.plot(level=1)
    view.hierarchicalGrid.globalRefine(2)
    


