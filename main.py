import sys
from pathlib import Path



from src.graphs.visualization import * 
from src.graphs.utils import *

from src.hyperbolicity.gromov import *


type = 'star'
G = create_graph(type,n=10)
G = add_edges(G,edges=[(1,2),(7,8),()])
pos = None
pos = draw_layout(G,type='spring', pos=pos) ## for the moment type is useless
draw_graph(G, pos) 

dist = compute_distance_nodes(G)

delta_max, delta_mean = compute_gromov_hyperbolicity(dist)

print(f"The gromov hyperbolicity (i.e. max of deltas) is: {delta_max}")
print(f"The mean of the values computed is: {delta_mean}")

delta_max, delta_mean = compute_gromov_hyperbolicity_not_optimized(dist)

print(f"The gromov hyperbolicity (i.e. max of deltas) is: {delta_max}")
print(f"The mean of the values computed is: {delta_mean}")