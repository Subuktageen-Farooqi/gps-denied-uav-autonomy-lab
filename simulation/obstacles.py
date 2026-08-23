def point_blocked(point, obstacles, inflation=0.0):
    x, y = point
    return any(x0-inflation <= x <= x1+inflation and y0-inflation <= y <= y1+inflation
               for x0, x1, y0, y1 in obstacles)

