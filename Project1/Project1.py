from matplotlib import pyplot as plt
import numpy as np
import matplotlib.tri as mtri

#Tasks 5-7

from dune.grid import structuredGrid, cartesianDomain
rect_gridView = structuredGrid([-0.5, -0.5], [2, 1], [10, 20])
#rect_gridView.plot()

from dune.alugrid import aluConformGrid
domain = cartesianDomain([-0.5, -0.5], [2, 1], [10, 20])
tri_gridView = aluConformGrid(domain)
#tri_gridView.plot()



#Tasks 8-9
import pygmsh

#Making triangular grid with hole
with pygmsh.occ.Geometry() as geom:
    geom.characteristic_length_min = 0.1
    geom.characteristic_length_max = 0.1
    
    grid = geom.add_rectangle([-0.5, -0.5, 0], 2.5, 1.5)
    hole = geom.add_disk([1, 0.5, 0.0], 0.1)
    domain = geom.boolean_difference(grid, hole)
    mesh = geom.generate_mesh()

import meshio

#meshio.write("mesh.vtk", mesh)

'''
# --- Extract triangle connectivity ---
points = mesh.points[:, :2]          # x,y only
triangles = mesh.cells_dict["triangle"]

# --- Tell matplotlib how triangles connect ---
triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)

# --- Plot ---
plt.triplot(triang)
plt.gca().set_aspect("equal")
plt.show()
'''



grid_points = mesh.points[:, :2]
grid_triangles = mesh.cells_dict["triangle"]

h_max = 0.0

for k in range(len(grid_triangles)):
    t = grid_triangles[k]          

    p0 = grid_points[t[0]]
    p1 = grid_points[t[1]]
    p2 = grid_points[t[2]]

    h_max = max(h_max, np.linalg.norm(p1 - p0))
    h_max = max(h_max, np.linalg.norm(p2 - p1))
    h_max = max(h_max, np.linalg.norm(p0 - p2))

print(h_max)


