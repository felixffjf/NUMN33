from matplotlib import pyplot as plt
import numpy as np
import matplotlib.tri as mtri

#Tasks 5-7

from dune.grid import structuredGrid, cartesianDomain
domain = cartesianDomain([-0.5, -0.5], [2, 1], [10, 20])
rect_gridView = structuredGrid([-0.5, -0.5], [2, 1], [10, 20])
rect_gridView.plot()

from dune.alugrid import aluConformGrid as leafGridView
tri_gridView = leafGridView(domain)
tri_gridView.plot()



#Tasks 8-9
import pygmsh

#Making triangular grid with hole
with pygmsh.occ.Geometry() as geom:
    geom.characteristic_length_min = 0.05
    geom.characteristic_length_max = 0.05
    
    grid = geom.add_rectangle([-0.5, -0.5, 0], 2.5, 1.5)
    hole = geom.add_disk([1, 0.5, 0.0], 0.1)
    domain = geom.boolean_difference(grid, hole)
    mesh = geom.generate_mesh()

points, cells = mesh.points[:,:2], mesh.cells_dict

dune_domain = {
    "vertices": points.astype("float"),
    "simplices": cells["triangle"]
}

gridView2D = leafGridView(dune_domain)
gridView2D.plot()


grid_points = mesh.points[:, :2]
grid_triangles = mesh.cells_dict["triangle"]


h_area = 0.0
for t in grid_triangles:
    p0, p1, p2 = grid_points[t]
    area = 0.5 * abs(np.cross(p1 - p0, p2 - p0))  
    h_area = max(h_area, area**(1/2))

print(h_area)
from dune.fem.utility import gridWidth
h_dune = gridWidth(gridView2D)
print(h_dune)


#Task 10-11




#Task 12-14

def uh(e,xhat,u):
    p0, p1, p2 = e[0], e[1], e[2]
    x0hat, x1hat = xhat[0], xhat[1]
    return u(p0)*(1-x0hat-x1hat) + u(p1)*x0hat + u(p2)*x1hat

