import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt

text = None
with open('casts.csv', 'r') as f:
    text = f.read()

text = text.decode('utf-8').encode('ascii', 'ignore')

G=nx.Graph()
films = text.split("\n")[0:200]
filmId = ''
filmsNumber = len(films)
for i in range (0, filmsNumber - 1):
    params = films[i].split(";")
    filmId = params[0]
    actor = params[2]
    G.add_node(actor)
    for j in range (i+1, filmsNumber - 1):
        params2 = films[j].split(";")
        if (filmId != params2[0]):
            break
        G.add_node(params2[2])
        G.add_edges_from([(actor, params2[2])])

pos = graphviz_layout(G, prog="fdp")

nx.draw(G, pos,
        labels={v:str(v) for v in G},
        cmap = plt.get_cmap("bwr"),
        node_color=[G.degree(v) for v in G],
        font_size=12
       )
plt.savefig("graph.png")
plt.show()

centralities = [nx.degree_centrality, nx.closeness_centrality,
nx.betweenness_centrality, nx.eigenvector_centrality]
region=220
for centrality in centralities:
    region+=1
    plt.subplot(region)
    plt.title(centrality.__name__)
    nx.draw(G, pos, labels={v:str(v) for v in G},
      cmap = plt.get_cmap("bwr"), node_color=[centrality(G)[k] for k in centrality(G)])
plt.savefig("centralities.png")
plt.show()

communities = {node:cid+1 for cid,community in enumerate(nx.k_clique_communities(G,3)) for node in community}

nx.draw(G, pos,
        labels={v:str(v) for v in G},
        cmap = plt.get_cmap("rainbow"),
        node_color=[communities[v] if v in communities else 0 for v in G])
plt.savefig("communities.png")
plt.show()