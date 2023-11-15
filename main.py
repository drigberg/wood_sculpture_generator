import json
import math
import random
import sys

from scipy.ndimage import gaussian_filter
from matplotlib import cm
import numpy as np
from PIL import Image

RANDOM_CONFIG = {
    "random": True,
    "branch": True,
    "color": True,
    "blur": 3,
    "width": 500,
    "height": 300,
    "density": 0.8,
    "maxCircleRadius": 0.2,
    "minCircleRadius": 0.1
}

EVEN_CONFIG = {
    "random": False,
    "branch": True,
    "color": True,
    "blur": 6,
    "width": 500,
    "height": 300,
    "density": 0.9,
    "maxCircleRadius": 0.2,
    "minCircleRadius": 0.1
}

class Circle:
    def __init__(self, x: int, y: int, radius: int):
        self.x = x
        self.y = y
        self.radius = radius

    def __str__(self) -> str:
        return f"x={self.x}, y={self.y}, radius={self.radius}"

class Sculptor:
    def __init__(self, config):
        self.config = config

    def create_field(self) -> list[list[float]]:
        self.field = [
            [0] * self.config['width']
            for i in range(self.config['height'])
        ]

        return self.field

    def apply_branches(self, circles: list[Circle]) -> None:
        self.apply_circles(circles)
        nodes = np.asarray([[circle.x, circle.y] for circle in circles])
        for i in range(len(nodes)):
            node = nodes[i]
            circle = circles[i]
            circle_coords = np.array([circle.x, circle.y])
            others = [n for n in nodes if n[0] != node[0] or n[1] != node[1]]
            deltas = others - node
            dist = np.einsum('ij,ij->i', deltas, deltas)
            closest = [others[x] for x in np.argpartition(dist, 4)[:4]]
            for other in closest:
                norm = np.linalg.norm(other-node)
                min_x = min(node[0], other[0])
                min_y = min(node[1], other[1])
                max_x = max(node[0], other[0])
                max_y = max(node[1], other[1])   
                for x in range(max(0, min_x), min(self.config["width"], max_x)):
                    for y in range(max(0, min_y), min(self.config["height"], max_y)):
                        d = np.linalg.norm(np.cross(other - node, node - [x, y])) / norm
                        self.field[y][x] += max(0, 1 - ((d / circle.radius) ** 2) * 20)

    def apply_circles(self, circles: list[Circle]) -> None:
        for circle in circles:
            for x in range(max(0, circle.x - circle.radius), min(self.config['width'], circle.x + circle.radius)):
                for y in range(max(0, circle.y - circle.radius), min(self.config['height'], circle.y + circle.radius)):
                    distance = math.sqrt((circle.x - x) ** 2 + (circle.y - y) ** 2)
                    if distance < circle.radius:
                        self.field[y][x] += 1 - distance / circle.radius

    def place_circles(self):
        min_side = min(self.config['width'], self.config['height'])
        num_circles = int(min_side * (self.config["minCircleRadius"] + self.config["maxCircleRadius"]) / 2 * self.config['density'] / 2)
        min_radius = int(min_side * self.config['minCircleRadius'])
        max_radius = int(min_side * self.config['maxCircleRadius'])

        print(f"Placing {num_circles} circles...")
        
        if num_circles < 6:
            raise Exception("Not enough circles: please adjust config")
        circles = []
        if self.config["random"] == True:
            for i in range(int(num_circles)):
                circle = Circle(
                    x=random.randint(0, self.config['width'] - 1),
                    y=random.randint(0, self.config['height'] - 1),
                    radius=random.randrange(min_radius, max_radius)
                )
                circles.append(circle)
        else:
            # See: https://stackoverflow.com/questions/27499139/how-can-i-set-a-minimum-distance-constraint-for-generating-points-with-numpy-ran
            # specify params
            n = num_circles
            shape = np.array([self.config["height"], self.config["width"]])
            sensitivity = self.config["density"]

            # compute grid shape based on number of points
            width_ratio = shape[1] / shape[0]
            num_y = np.int32(np.sqrt(n / width_ratio)) + 1
            num_x = np.int32(n / num_y) + 1

            # create regularly spaced neurons
            x = np.linspace(0., shape[1]-1, num_x, dtype=np.float32)
            y = np.linspace(0., shape[0]-1, num_y, dtype=np.float32)
            coords = np.stack(np.meshgrid(x, y), -1).reshape(-1,2)

            # compute spacing
            init_dist = np.min((x[1]-x[0], y[1]-y[0]))
            min_dist = init_dist * (1 - sensitivity)

            assert init_dist >= min_dist

            # perturb points
            max_movement = (init_dist - min_dist)/2
            noise = np.random.uniform(
                low=-max_movement,
                high=max_movement,
                size=(len(coords), 2))
            coords += noise
            for coord in coords:
                circle = Circle(
                    x=int(min(coord[0], self.config["width"] - 1)),
                    y=int(min(coord[1], self.config["height"] - 1)),
                    radius=random.randrange(min_radius, max_radius)
                )
                circles.append(circle)
        self.apply_branches(circles)
       
        
    def soften(self):
        self.field = np.clip(self.field, 0, 2)
        self.field = (self.field - np.min(self.field))/(np.max(self.field) - np.min(self.field))
        self.field = gaussian_filter(self.field, sigma=self.config['blur'])
        self.field = (self.field - np.min(self.field))/(np.max(self.field) - np.min(self.field))


    def generate(self):
        self.create_field()
        self.place_circles()
        self.soften()
        return self.field
    
    def print(self):
        if self.config['color'] == True:
            array = np.uint8(cm.gist_earth(self.field) * 255)
        else:
            pixels = [[min(255, int(p * 255)) for p in row] for row in self.field]
            array = np.array(pixels, dtype=np.uint8)
            
        image = Image.fromarray(array)
        image.save('output/temp.png')

if __name__ == '__main__':
    print("Running...")

    config = RANDOM_CONFIG
    if len(sys.argv) > 1:
        if sys.argv[1] == "RANDOM":
            config = RANDOM_CONFIG
        elif sys.argv[1] == "EVEN":
            config = EVEN_CONFIG
        else:
            raise Exception(f"Unrecognized config: {sys.argv[1]}")

    sculptor = Sculptor(config)
    sculptor.generate()
    sculptor.print()
    print("Done!")