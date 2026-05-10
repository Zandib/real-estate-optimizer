from shapely.geometry import Polygon

class QuadPoint:
    def __init__(self, x, y, value):
        self.x, self.y, self.value = x, y, value

class Boundary:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    def contains(self, point):
        return (self.x - self.w <= point.x <= self.x + self.w and
                self.y - self.h <= point.y <= self.y + self.h)

    def get_polygon(self):
        """Retorna a geometria do Boundary como um Polygon."""
        min_x = self.x - self.w
        max_x = self.x + self.w
        min_y = self.y - self.h
        max_y = self.y + self.h
        return Polygon([
            (min_x, min_y),
            (min_x, max_y),
            (max_x, max_y),
            (max_x, min_y),
            (min_x, min_y)
        ])

class QuadTree:
    def __init__(self, boundary, capacity, depth=0, uid="0"):
        self.boundary = boundary
        self.capacity = capacity
        self.points = []
        self.divided = False
        self.depth = depth
        self.uid = uid # Identificador único do quadrante

    def subdivide(self):
        x, y, w, h = self.boundary.x, self.boundary.y, self.boundary.w, self.boundary.h

        # Criando sub-quadrantes com IDs hierárquicos (ex: 0.1, 0.2...)
        self.northwest = QuadTree(Boundary(x - w/2, y + h/2, w/2, h/2), self.capacity, self.depth + 1, f"{self.uid}.1")
        self.northeast = QuadTree(Boundary(x + w/2, y + h/2, w/2, h/2), self.capacity, self.depth + 1, f"{self.uid}.2")
        self.southwest = QuadTree(Boundary(x - w/2, y - h/2, w/2, h/2), self.capacity, self.depth + 1, f"{self.uid}.3")
        self.southeast = QuadTree(Boundary(x + w/2, y - h/2, w/2, h/2), self.capacity, self.depth + 1, f"{self.uid}.4")
        self.divided = True

    def insert(self, point):
        if not self.boundary.contains(point):
            return False

        if len(self.points) < self.capacity and not self.divided:
            self.points.append(point)
            return True
        else:
            if not self.divided:
                self.subdivide()
                # Move os pontos existentes para os novos quadrantes
                old_points = self.points
                self.points = []
                for p in old_points:
                    self._insert_to_children(p)

            return self._insert_to_children(point)

    def _insert_to_children(self, point):
        return (self.northwest.insert(point) or self.northeast.insert(point) or
                self.southwest.insert(point) or self.southeast.insert(point))

    def get_quadrant_id(self, point):
        """Retorna o UID do menor quadrante (folha) que contém o ponto."""
        if not self.boundary.contains(point):
            return None

        if not self.divided:
            return self.uid

        # Busca recursiva nos filhos
        for child in [self.northwest, self.northeast, self.southwest, self.southeast]:
            res = child.get_quadrant_id(point)
            if res: return res
        return self.uid

    def get_quadrant_geometry(self, quad_uid):
        """Retorna a geometria (Polygon) de um quadrante dado o seu UID."""
        if self.uid == quad_uid:
            return self.boundary.get_polygon()

        if not self.divided:
            return None

        for child in [self.northwest, self.northeast, self.southwest, self.southeast]:
            geometry = child.get_quadrant_geometry(quad_uid)
            if geometry:
                return geometry
        return None

def get_params(lat_t, lon_t):
    return sum(lon_t)/2, sum(lat_t)/2, abs(lon_t[0]-lon_t[1])/2, abs(lat_t[0]-lat_t[1])/2
