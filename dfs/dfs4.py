import collections

# Grafo do sétimo exemplo do dataset
edges = [
    ("cfcd208495", "eccbc87e4b"),
    ("cfcd208495", "a87ff679a2"),
    ("cfcd208495", "eccbc87e4b"),
    ("cfcd208495", "45c48cce2e"),
    ("c4ca4238a0", "45c48cce2e"),
    ("c4ca4238a0", "45c48cce2e"),
    ("c4ca4238a0", "cfcd208495"),
    ("c4ca4238a0", "c4ca4238a0"),
    ("c81e728d9d", "d3d9446802"),
    ("c81e728d9d", "cfcd208495"),
    ("c81e728d9d", "e4da3b7fbb"),
    ("c81e728d9d", "cfcd208495"),
    ("eccbc87e4b", "c9f0f895fb"),
    ("eccbc87e4b", "eccbc87e4b"),
    ("eccbc87e4b", "e4da3b7fbb"),
    ("eccbc87e4b", "c9f0f895fb"),
    ("a87ff679a2", "d3d9446802"),
    ("a87ff679a2", "d3d9446802"),
    ("a87ff679a2", "cfcd208495"),
    ("a87ff679a2", "c9f0f895fb"),
    ("e4da3b7fbb", "eccbc87e4b"),
    ("e4da3b7fbb", "c4ca4238a0"),
    ("e4da3b7fbb", "8f14e45fce"),
    ("e4da3b7fbb", "c81e728d9d"),
    ("1679091c5a", "45c48cce2e"),
    ("1679091c5a", "c4ca4238a0"),
    ("1679091c5a", "c9f0f895fb"),
    ("1679091c5a", "eccbc87e4b"),
    ("8f14e45fce", "cfcd208495"),
    ("8f14e45fce", "eccbc87e4b"),
    ("8f14e45fce", "c4ca4238a0"),
    ("8f14e45fce", "45c48cce2e"),
    ("c9f0f895fb", "eccbc87e4b"),
    ("c9f0f895fb", "1679091c5a"),
    ("c9f0f895fb", "c81e728d9d"),
    ("c9f0f895fb", "45c48cce2e"),
    ("45c48cce2e", "d3d9446802"),
    ("45c48cce2e", "c4ca4238a0"),
    ("45c48cce2e", "c81e728d9d"),
    ("45c48cce2e", "1679091c5a"),
    ("d3d9446802", "eccbc87e4b"),
    ("d3d9446802", "e4da3b7fbb"),
    ("d3d9446802", "eccbc87e4b"),
    ("d3d9446802", "45c48cce2e"),
]

graph       = collections.defaultdict(list)   # filhos
rev_graph   = collections.defaultdict(list)   # pais

for u, v in edges:
    graph[u].append(v)
    rev_graph[v].append(u)

# Busca em profundidade 
def dfs_path(start, goal, graph):
    stack = [(start, None)]    # (nó, pai)
    seen  = {}                 # nó -> pai

    while stack:
        node, parent = stack.pop()
        if node in seen:
            continue
        seen[node] = parent
        if node == goal:
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = seen[cur]
            return list(reversed(path)), seen
        for nbr in reversed(graph.get(node, [])):
            stack.append((nbr, node))
    return None, seen

# Funções utilizadas
def pais_de(node, rev_graph):
    return list(dict.fromkeys(rev_graph.get(node, []))) 

def filhos_de(node, graph):
    return list(dict.fromkeys(graph.get(node, [])))

def pertence_ao_grafo(node, came_from):
    return node in came_from

# Testes
start = "cfcd208495"

print(" Pais de um nó")
for n in ("45c48cce2e", "c9f0f895fb"):
    print(f"Pais de '{n}':", pais_de(n, rev_graph))

print("\n Filhos de um nó")
for n in ("cfcd208495", "45c48cce2e"):
    print(f"Filhos de '{n}':", filhos_de(n, graph))

print("\n Testa se o nó está no grafo")
testa = "d3d9446802"
print(f"'{testa}' pertence? ->",
      pertence_ao_grafo(testa, dfs_path(start, testa, graph)[1]))

inexistente = "cfcd208495"
print(f"'{inexistente}' pertence? ->",
      pertence_ao_grafo(inexistente, dfs_path(start, inexistente, graph)[1]))
