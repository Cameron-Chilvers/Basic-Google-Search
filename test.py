from collections import defaultdict

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
    
    def addEdge(self, u, v):
        self.graph[u].append(v)
    
    def DFSUtil(self, v, visited):
        visited.add(v)
        path = [v]
        
        for neighbour in self.graph[v]:
            if neighbour not in visited:
                path.extend(self.DFSUtil(neighbour, visited))
        
        return path
    
    def allDFSPathsUtil(self, v, visited, path, all_paths):
        visited.add(v)
        path.append(v)
        
        if len(path) == len(self.graph):
            all_paths.append(path[:])
        
        for neighbour in self.graph[v]:
            if neighbour not in visited:
                self.allDFSPathsUtil(neighbour, visited, path, all_paths)
        
        path.pop()
        visited.remove(v)
    
    def allDFSPaths(self, v):
        visited = set()
        path = []
        all_paths = []
        
        self.allDFSPathsUtil(v, visited, path, all_paths)
        
        return all_paths

if __name__ == "__main__":
    g = Graph()
    # A
    g.addEdge(0, 1)
    g.addEdge(0, 2)
    g.addEdge(0, 5)
    g.addEdge(0, 6)

    # B
    g.addEdge(1, 0)
    g.addEdge(1, 3)
    g.addEdge(1, 4)
    g.addEdge(1, 6)

    # C
    g.addEdge(2, 0)
    g.addEdge(2, 3)
    
    # D
    g.addEdge(3, 1)
    g.addEdge(3, 2)
    g.addEdge(3, 4)

    # E
    g.addEdge(4, 0)
    g.addEdge(4, 1)
    g.addEdge(4, 3)
    g.addEdge(4, 5)

    # F
    g.addEdge(5, 0)
    g.addEdge(5, 4)
    g.addEdge(5, 6)

    # G
    g.addEdge(6, 1)
    g.addEdge(6, 5)
    
    dfs_orders = g.allDFSPaths(0)
    for order in dfs_orders:
        print(order)
