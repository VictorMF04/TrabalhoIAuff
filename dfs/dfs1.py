import collections

# Grafo do primeiro exemplo do dataset
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
for n in ("eccbc87e4b", "c9f0f895fb"):
    print(f"Pais de '{n}':", pais_de(n, rev_graph))

print("\n Filhos de um nó")
for n in ("1679091c5a", "c9f0f895fb"):
    print(f"Filhos de '{n}':", filhos_de(n, graph))

print("\n Testa se o nó está no grafo")
testa1 = "c4ca4238a0"
print(f"'{testa1}' pertence? ->",
      pertence_ao_grafo(testa1, dfs_path(start, testa1, graph)[1]))

testa2 = "deadbeef"
print(f"'{testa2}' pertence? ->",
      pertence_ao_grafo(testa2, dfs_path(start, testa2, graph)[1]))
