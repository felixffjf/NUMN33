from matplotlib import pyplot as plt
import numpy as np

from dune.grid import structuredGrid, cartesianDomain
rect_gridView = structuredGrid([-0.5, -0.5], [2, 1], [10, 20])
rect_gridView.plot()

from dune.alugrid import aluConformGrid
domain = cartesianDomain([-0.5, -0.5], [2, 1], [10, 20])
tri_gridView = aluConformGrid(domain)
tri_gridView.plot()