import math
import random

from scipy.ndimage import gaussian_filter
from matplotlib import cm
import numpy as np
from PIL import Image

CONFIG = {
    "width": 500,
    "height": 300,
    "blur": 3,
    "density": 5,
    "maxCircleRadius": 0.03,
    "minCircleRadius": 0.02
}


class Circle:
    def __init__(self, x: int, y: int, radius: int):
        self.x = x
        self.y = y
        self.radius = radius

    def distance(self, other) -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

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
        num_circles = int(min_side * (1 - (self.config["minCircleRadius"] + self.config["maxCircleRadius"])) * self.config['density'] / 4)
        min_radius = int(min_side * self.config['minCircleRadius'])
        max_radius = int(min_side * self.config['maxCircleRadius'])

        print(f"Placing {num_circles} circles...")
        
        if num_circles < 6:
            raise Exception("Not enough circles: please adjust config")
        
        circles = []
        for i in range(int(num_circles)):
            circle = Circle(
                x=random.randint(0, self.config['width'] - 1),
                y=random.randint(0, self.config['height'] - 1),
                radius=random.randrange(min_radius, max_radius)
            )
            attempts = 0
            while any([c.distance(circle) < max(c.radius, circle.radius) for c in circles]):
                circle = Circle(
                    x=random.randint(0, self.config['width'] - 1),
                    y=random.randint(0, self.config['height'] - 1),
                    radius=random.randrange(min_radius, max_radius)
                )
                attempts += 1
                if attempts > 10:
                    break
            circles.append(circle)
        self.apply_branches(circles)
       
    def blur(self):
        self.field = gaussian_filter(self.field, sigma=self.config['blur'])

    def sharpen(self):
        # See: https://scipy-lectures.org/advanced/image_processing/auto_examples/plot_sharpen.html
        self.field = self.field + 2 * (self.field - gaussian_filter(self.field, sigma=self.config['blur']))

    def normalize(self):
        self.field = (self.field - np.min(self.field)) / (np.max(self.field) - np.min(self.field))

    def stretch(self):
        self.field = (np.array(self.field) * 100) ** 2

    def clip(self):
        self.field = np.clip(self.field, 0, 1)

    def floor(self):
        self.field = [[v / 2 if v <= 0.2 else v for v in row] for row in self.field]
        self.field = [[1 if v > 0.2 else v for v in row] for row in self.field]

    def soften(self):
        self.clip()
        self.blur()
        self.normalize()
        self.floor()
        self.blur()
        self.normalize()



    def generate(self):
        self.create_field()
        self.place_circles()
        self.soften()
        return self.field
    
    def print(self):
        pixels = [[min(255, int(p * 255)) for p in row] for row in self.field]
        array = np.array(pixels, dtype=np.uint8) 
        image = Image.fromarray(array)
        image.save('output/temp.png')

if __name__ == '__main__':
    print("Running...")

    sculptor = Sculptor(CONFIG)
    sculptor.generate()
    sculptor.print()
    print("Done!")