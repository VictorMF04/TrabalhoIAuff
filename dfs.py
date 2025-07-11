import collections

# Grafo de exemplo tirado do dataset
edges = [
    ("cfcd208495","eccbc87e4b"),
    ("cfcd208495","c4ca4238a0"),
    ("cfcd208495","1679091c5a"),
    ("cfcd208495","a87ff679a2"),
    ("c4ca4238a0","c9f0f895fb"),
    ("c4ca4238a0","1679091c5a"),
    ("c4ca4238a0","a87ff679a2"),
    ("c81e728d9d","c4ca4238a0"),
    ("c81e728d9d","a87ff679a2"),
    ("eccbc87e4b","8f14e45fce"),
    ("eccbc87e4b","cfcd208495"),
    ("eccbc87e4b","c9f0f895fb"),
    ("eccbc87e4b","a87ff679a2"),
    ("eccbc87e4b","e4da3b7fbb"),
    ("a87ff679a2","c4ca4238a0"),
    ("a87ff679a2","8f14e45fce"),
    ("a87ff679a2","c81e728d9d"),
    ("a87ff679a2","1679091c5a"),
    ("e4da3b7fbb","eccbc87e4b"),
    ("e4da3b7fbb","8f14e45fce"),
    ("e4da3b7fbb","1679091c5a"),
    ("e4da3b7fbb","c4ca4238a0"),
    ("1679091c5a","a87ff679a2"),
    ("1679091c5a","c9f0f895fb"),
    ("1679091c5a","c4ca4238a0"),
    ("1679091c5a","c81e728d9d"),
    ("8f14e45fce","a87ff679a2"),
    ("8f14e45fce","e4da3b7fbb"),
    ("8f14e45fce","c81e728d9d"),
    ("8f14e45fce","c4ca4238a0"),
    ("c9f0f895fb","1679091c5a"),
    ("c9f0f895fb","cfcd208495"),
]

graph = collections.defaultdict(list)
for u, v in edges:
    graph[u].append(v)

# Busca em profundidade
def dfs_path(start, goal, graph):
    """
    Retorna (path, came_from) se goal alcançável a partir de start.
    Se não houver caminho, path = None.
    """
    stack = [(start, None)]          
    came_from = {}                   

    while stack:
        node, parent = stack.pop()
        if node in came_from:        
            continue
        came_from[node] = parent

        if node == goal:
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            return list(reversed(path)), came_from

        
        for nbr in reversed(graph.get(node, [])):    
            stack.append((nbr, node))


    return None, came_from

# Funções utilizadas
def pai_de(node, came_from):
    return came_from.get(node)

def filhos_de(node, graph):
    return graph.get(node, [])

def pertence_ao_grafo(node, came_from):
    return node in came_from

# Testes
start = "cfcd208495"  

print("Pai de um nó")

_, came = dfs_path(start, "eccbc87e4b", graph)
print(f"Pai de 'eccbc87e4b': {pai_de('eccbc87e4b', came)}")

_, came = dfs_path(start, "8f14e45fce", graph)
print(f"Pai de '8f14e45fce': {pai_de('8f14e45fce', came)}")


print("\nFilhos de um nó")
no1 = "cfcd208495"
no2 = "1679091c5a"
print(f"Filhos de '{no1}':", filhos_de(no1, graph))
print(f"Filhos de '{no2}':", filhos_de(no2, graph))


print("\nTesta se o nó está no grafo")
pertence = "c4ca4238a0"
path, came = dfs_path(start, pertence, graph)
print(f"'{pertence}' pertence? ->", pertence_ao_grafo(pertence, came),
      "| Caminho:", path)
inexistente = "deadbeef"

path, came = dfs_path(start, inexistente, graph)
print(f"'{inexistente}' pertence? ->", pertence_ao_grafo(inexistente, came),
      "| Caminho :", path)
