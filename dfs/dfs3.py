import collections

# Grafo do sexto exemplo do dataset
edges = [
    ("cfcd208495", "45c48cce2e"),
    ("c4ca4238a0", "c74d97b01e"),
    ("c81e728d9d", "1c383cd30b"),
    ("eccbc87e4b", "d645920e39"),
    ("a87ff679a2", "e369853df7"),
    ("e4da3b7fbb", "c74d97b01e"),
    ("1679091c5a", "70efdf2ec9"),
    ("8f14e45fce", "d645920e39"),
    ("c9f0f895fb", "aab3238922"),
    ("45c48cce2e", "1f0e3dad99"),
    ("d3d9446802", "d645920e39"),
    ("6512bd43d9", "34173cb38f"),
    ("c20ad4d76f", "eccbc87e4b"),
    ("c51ce410c1", "182be0c5cd"),
    ("aab3238922", "a1d0c6e83f"),
    ("9bf31c7ff0", "a5771bce93"),
    ("c74d97b01e", "70efdf2ec9"),
    ("70efdf2ec9", "aab3238922"),
    ("6f4922f455", "c4ca4238a0"),
    ("1f0e3dad99", "f7177163c8"),
    ("98f1370821", "c20ad4d76f"),
    ("3c59dc048e", "aab3238922"),
    ("b6d767d2f8", "4e732ced34"),
    ("37693cfc74", "e369853df7"),
    ("1ff1de7740", "33e75ff09d"),
    ("8e296a067a", "3c59dc048e"),
    ("4e732ced34", "c81e728d9d"),
    ("02e74f10e0", "6ea9ab1baa"),
    ("33e75ff09d", "c74d97b01e"),
    ("6ea9ab1baa", "98f1370821"),
    ("34173cb38f", "37693cfc74"),
    ("c16a5320fa", "33e75ff09d"),
    ("6364d3f0f4", "3c59dc048e"),
    ("182be0c5cd", "a87ff679a2"),
    ("e369853df7", "8e296a067a"),
    ("1c383cd30b", "a5771bce93"),
    ("19ca14e7ea", "d3d9446802"),
    ("a5bfc9e079", "1c383cd30b"),
    ("a5771bce93", "98f1370821"),
    ("d67d8ab4f4", "d3d9446802"),
    ("d645920e39", "9bf31c7ff0"),
    ("3416a75f4c", "c9f0f895fb"),
    ("a1d0c6e83f", "182be0c5cd"),
    ("17e62166fc", "182be0c5cd"),
    ("f7177163c8", "c16a5320fa"),
    ("6c8349cc72", "17e62166fc"),
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
for n in ("3416a75f4c", "aab3238922"):
    print(f"Pais de '{n}':", pais_de(n, rev_graph))

print("\n Filhos de um nó")
for n in ("34173cb38f", "c74d97b01e"):
    print(f"Filhos de '{n}':", filhos_de(n, graph))

print("\n Testa se o nó está no grafo")
testa1 = "e369853df7"
print(f"'{testa1}' pertence? ->",
      pertence_ao_grafo(testa1, dfs_path(start, testa1, graph)[1]))

testa2 = "8f14e45fce"
print(f"'{testa2}' pertence? ->",
      pertence_ao_grafo(testa2, dfs_path(start, testa2, graph)[1]))
