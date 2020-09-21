class Path_Calculator:
    def __init__(self, cost, heuristic, start_point, goals):
        self.cost_matrix = cost
        self.heuristic_list = heuristic
        self.start_point = start_point
        self.goals = goals
        # path_list -> [[path_cost1,[1,2,3]],[path_cost2,[1,2,4,5]].....]
        self.ucs_path = {"path_list": [],
                         "visited": []}
        self.dfs_path = {"path_list": [],
                         "visited": set()}
        # path_list -> [[path_cost + heuristc,[1,2,4,6]],......]
        self.astar_path = {"path_list": [],
                           "visited": []}
        self.temp = start_point

    def find_dfs_paths(self,node,visited):
        paths = self.cost_matrix[node]
        temp = []
        for i, cost in enumerate(paths):
            if cost>0 and i not in visited:
                temp.append(i)
        temp.reverse()
        return temp

    def calculate_dfs_path(self):
        self.reset_temp()

        stack = [(self.start_point, [self.start_point])]
        while stack:
            (node, self.dfs_path["path_list"]) = stack.pop()
            if node not in self.dfs_path["visited"]:
                if node in self.goals:
                    return self.dfs_path["path_list"]
                self.dfs_path["visited"].add(node)
                path_list = self.find_dfs_paths(node,self.dfs_path["visited"])
                for neighbour in path_list:
                    stack.append((neighbour, self.dfs_path["path_list"] + [neighbour]))
        return stack

    def calculate_ucs_path(self):
        self.reset_temp()

        if self.start_point in self.goals:
            return [self.start_point]

        self.ucs_path["visited"].append(self.temp)
        paths = self.cost_matrix[self.temp]
        possible_paths = [[paths[i], i]
                          for i in range(1, len(paths))
                          if paths[i] != 0 and paths[i] != -1]
        possible_paths.sort()
        for path in possible_paths:
            self.ucs_path["path_list"].append([path[0], [self.temp, path[1]]])

        while self.ucs_path["path_list"]:
            least_cost = self.ucs_path["path_list"][0][0]
            if self.ucs_path["path_list"][0][1][-1] not in self.ucs_path["visited"]:
                self.temp = self.ucs_path["path_list"][0][1]
                self.ucs_path["path_list"].pop(0)
                self.ucs_path["visited"].append(self.temp[-1])

                if self.temp[-1] in self.goals:
                    return self.temp
                paths = self.cost_matrix[self.temp[-1]]
                possible_paths = [[paths[i], i]
                                  for i in range(1, len(paths))
                                  if paths[i] != 0 and paths[i] != -1]
                if possible_paths:
                    for path in possible_paths:
                        self.ucs_path["path_list"].append(
                            [path[0] + least_cost, self.temp + [path[1]]])
                    self.ucs_path["path_list"].sort()

            else:
                self.ucs_path["path_list"].pop(0)

        self.temp = self.start_point
        return []

    def calculate_astar_path(self):
        self.reset_temp()

        if self.start_point in self.goals:
            return [self.start_point]

        self.astar_path["visited"].append(self.temp)
        paths = self.cost_matrix[self.temp]
        possible_paths = [[paths[i]+self.heuristic_list[i],i]
                          for i in range(1,len(paths)) if paths[i]!=0 and paths[i]!=-1]
        possible_paths.sort()
        for path in possible_paths:
            self.astar_path["path_list"].append([path[0],[self.temp,path[1]]])

        while self.astar_path["path_list"]:
            least_cost = self.astar_path["path_list"][0][0]
            if self.astar_path["path_list"][0][1][-1] not in self.astar_path["visited"]:
                self.temp = self.astar_path["path_list"][0][1]
                self.astar_path["path_list"].pop(0)
                self.astar_path["visited"].append(self.temp[-1])

                if self.temp[-1] in self.goals:
                    return self.temp
                else:
                    paths = self.cost_matrix[self.temp[-1]]
                    possible_paths = [[paths[i]+self.heuristic_list[i],i]
                                      for i in range(1,len(paths)) if paths[i]!=0 and paths[i]!=-1]
                    if possible_paths:
                        for path in possible_paths:
                            self.astar_path["path_list"].append([path[0] -
                                                                 self.heuristic_list[self.temp[-1]] +
                                                                 least_cost,
                                                                 self.temp + [path[1]]])
                        self.astar_path["path_list"].sort()

            else:
                self.astar_path["path_list"].pop(0)

        self.temp = self.start_point
        return []
    def reset_temp(self):
        self.temp = self.start_point

def tri_traversal(cost, heuristic, start, goals):
    path_obj = Path_Calculator(cost, heuristic, start, goals)

    ucs_path = path_obj.calculate_ucs_path()
    path_obj.reset_temp()
    astar_path = path_obj.calculate_astar_path()
    path_obj.reset_temp()
    dfs_path = path_obj.calculate_dfs_path()

    return [dfs_path, ucs_path, astar_path]

def DFS_Traversal(cost, start, goals):
    path_obj = Path_Calculator(cost, [], start, goals)

    res = path_obj.calculate_dfs_path()
    return res

def UCS_Traversal(cost, start, goals):
    path_obj = Path_Calculator(cost, [], start, goals)

    res = path_obj.calculate_ucs_path()
    return res

def A_star_Traversal(cost, heuristic, start, goals):
    path_obj = Path_Calculator(cost, heuristic, start, goals)

    res = path_obj.calculate_astar_path()
    return res
