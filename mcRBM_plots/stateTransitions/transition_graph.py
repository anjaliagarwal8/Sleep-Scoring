import networkx as nx
from fa2 import ForceAtlas2
import matplotlib.pyplot as plt
import numpy as np

forceatlas2 = ForceAtlas2(
                        # Behavior alternatives
                        outboundAttractionDistribution=True,  # Dissuade hubs
                        linLogMode=False,  # NOT IMPLEMENTED
                        adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                        edgeWeightInfluence=1.0,

                        # Performance
                        jitterTolerance=1.0,  # Tolerance
                        barnesHutOptimize=True,
                        barnesHutTheta=1.2,
                        multiThreaded=False,  # NOT IMPLEMENTED

                        # Tuning
                        scalingRatio=2.0,
                        strongGravityMode=False,
                        gravity=1.0,

                        # Log
                        verbose=True)

G = nx.Graph()
latentStates = np.linspace(1,318,318)
G.add_nodes_from(latentStates)


print("Number of nodes: ",G.number_of_nodes)
print("Number of edges: ",G.number_of_edges)
print("Nodes: ",list(G.nodes))
print("Edges: ",list(G.edges))

positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=2000)

nx.draw_networkx_nodes(G, positions, node_size=200, with_labels=True, node_color="blue", alpha=1)
nx.draw_networkx_edges(G, positions, edge_color="blue", alpha=1)
nx.draw_networkx_labels(G, positions, font_weight = 'bold')
plt.axis('off')
plt.show()