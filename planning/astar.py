import heapq
import math
import numpy as np


class AStarPlanner:
    def __init__(self, width, height, resolution=1.0, obstacles=(), inflation=1.0):
        self.resolution=resolution
        self.nx, self.ny=int(width/resolution)+1, int(height/resolution)+1
        self.blocked=set()
        for x0,x1,y0,y1 in obstacles:
            for ix in range(max(0,int((x0-inflation)/resolution)), min(self.nx,int((x1+inflation)/resolution)+1)):
                for iy in range(max(0,int((y0-inflation)/resolution)), min(self.ny,int((y1+inflation)/resolution)+1)):
                    self.blocked.add((ix,iy))

    def _grid(self,p): return tuple(np.clip(np.round(np.asarray(p)/self.resolution).astype(int), [0,0], [self.nx-1,self.ny-1]))
    def _world(self,p): return np.asarray(p,float)*self.resolution

    def plan(self, start, goal):
        s,g=self._grid(start),self._grid(goal)
        if g in self.blocked:
            free=[(math.hypot(x-g[0],y-g[1]),(x,y)) for x in range(self.nx) for y in range(self.ny) if (x,y) not in self.blocked]
            if not free: return []
            g=min(free)[1]
        moves=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        queue=[(0,s)]; came={}; cost={s:0.}
        while queue:
            _,cur=heapq.heappop(queue)
            if cur==g: break
            for dx,dy in moves:
                nxt=(cur[0]+dx,cur[1]+dy)
                if not (0<=nxt[0]<self.nx and 0<=nxt[1]<self.ny) or nxt in self.blocked: continue
                if dx and dy and ((cur[0]+dx,cur[1]) in self.blocked or (cur[0],cur[1]+dy) in self.blocked): continue
                nc=cost[cur]+math.hypot(dx,dy)
                if nc < cost.get(nxt,float('inf')):
                    cost[nxt]=nc; came[nxt]=cur
                    h=math.hypot(nxt[0]-g[0],nxt[1]-g[1]); heapq.heappush(queue,(nc+h,nxt))
        if g not in cost: return []
        path=[g]
        while path[-1]!=s: path.append(came[path[-1]])
        return [self._world(p) for p in reversed(path)]

