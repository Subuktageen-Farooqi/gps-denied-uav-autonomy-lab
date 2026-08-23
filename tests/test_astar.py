from planning.astar import AStarPlanner


def test_path_avoids_obstacle():
    p=AStarPlanner(20,20,1,[(8,12,0,15)],.5).plan((2,2),(18,2))
    assert p and all(not (7.5<=x<=12.5 and 0<=y<=15.5) for x,y in p)


def test_direct_path_exists():
    assert AStarPlanner(10,10).plan((1,1),(9,9))

