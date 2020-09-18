
class Path_Calculator:
    def __init__(self,cost,heuristic,start_point,goals):
        self.cost_matrix = cost
        self.heuristic_list = heuristic
        self.start_point = start_point
        self.goals = goals
        self.ucs_path = {"path_list":[],"visited":[]}# path_list -> [[path_cost1,[1,2,3]],[path_cost2,[1,2,4,5]].....]
        self.dfs_path = {}
        self.astar_path = {"path_list":[],"visited":[]}# path_list -> [[path_cost + heuristc,[1,2,4,6]],......]
        self.temp = start_point


    def calculate_ucs_path(self):
        if self.start_point in self.goals:
            retrun [self.start_point]

        self.ucs_path["visited"].append(self.temp)
        paths = self.cost_matrix[self.temp]
        possible_paths = [[paths[i],i] for i in range(1,len(paths)) if paths[i]!=0 and paths[i]!=-1]
        possible_paths.sort()
        for path in possible_paths:
            self.ucs_path["path_list"].append([path[0],[self.temp,path[1]]])

        while(self.ucs_path["path_list"]):
            least_cost = self.ucs_path["path_list"][0][0]
            if self.ucs_path["path_list"][0][1][-1] not in self.ucs_path["visited"]:
                self.temp = self.ucs_path["path_list"][0][1]
                self.ucs_path["path_list"].pop(0)
                self.ucs_path["visited"].append(self.temp[-1])
                
                if(self.temp[-1] in self.goals):
                    return self.temp
                else:
                    paths = self.cost_matrix[self.temp[-1]]
                    possible_paths = [[paths[i],i] for i in range(1,len(paths)) if paths[i]!=0 and paths[i]!=-1]
                    if possible_paths:
                        for path in possible_paths:
                            self.ucs_path["path_list"].append([path[0]+least_cost,self.temp + [path[1]]])
                        self.ucs_path["path_list"].sort()

            else:
                self.ucs_path["path_list"].pop(0)
            
        self.temp = self.start_point
        return []
            
    def calculate_astar_path(self):
        if self.start_point in self.goals:
            retrun [self.start_point]

        self.astar_path["visited"].append(self.temp)
        paths = self.cost_matrix[self.temp]
        possible_paths = [[paths[i]+self.heuristic_list[i-1],i] for i in range(1,len(paths)) if paths[i]!=0 and paths[i]!=-1]
        possible_paths.sort()
        for path in possible_paths:
            self.astar_path["path_list"].append([path[0],[self.temp,path[1]]])

        while(self.ucs_path["path_list"]):
            least_cost = self.astar_path["path_list"][0][0]
            if self.astar_path["path_list"][0][1][-1] not in self.astar_path["visited"]:
                self.temp = self.astar_path["path_list"][0][1]
                self.astar_path["path_list"].pop(0)
                self.astar_path["visited"].append(self.temp[-1])
                
                if(self.temp[-1] in self.goals):
                    return self.temp
                else:
                    paths = self.cost_matrix[self.temp[-1]]
                    possible_paths = [[paths[i]+self.heuristic_list[i-1],i] for i in range(1,len(paths)) if paths[i]!=0 and paths[i]!=-1]
                    if possible_paths:
                        for path in possible_paths:
                            self.astar_path["path_list"].append([path[0]+least_cost,self.temp + [path[1]]])
                        self.astar_path["path_list"].sort()

            else:
                self.astar_path["path_list"].pop(0)
            
        self.temp = self.start_point
        return []

    def reset_temp(self):
        self.temp = self.start_point

if __name__=="__main__":
    cost = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 5, 9, -1, 6, -1, -1, -1, -1, -1],
            [0, -1, 0, 3, -1, -1, 9, -1, -1, -1, -1], 
            [0, -1, 2, 0, 1, -1, -1, -1, -1, -1, -1],
            [0, 6, -1, -1, 0, -1, -1, 5, 7, -1, -1],
            [0, -1, -1, -1, 2, 0, -1, -1, -1, 2, -1],
            [0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1],
            [0, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1],
            [0, -1, -1, -1, -1, 2, -1, -1, 0, -1, 8],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, 0, 7],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]]
    
    heuristic = [0, 5, 7, 3, 4, 6, 0, 0, 6, 5, 0]
    path_obj = Path_Calculator(cost,heuristic,1,[6,7,10])
    res = path_obj.calculate_ucs_path()
    print(res)
    path_obj.reset_temp()
    res1 = path_obj.calculate_astar_path()
    print(res1)
