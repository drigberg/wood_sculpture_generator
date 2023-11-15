import json
import math
import random
import sys

from scipy.ndimage.filters import gaussian_filter
from matplotlib import cm
import numpy as np
from PIL import Image

RANDOM_CONFIG = {
    "distribution": "RANDOM",
    "color": True,
    "blur": 10,
    "width": 500,
    "height": 300,
    "density": 0.2,
    "maxCircleRadius": 0.2,
    "minCircleRadius": 0.1
}

EVEN_CONFIG = {
    "distribution": "RANDOM",
    "blur": 15,
    "color": True,
    "width": 500,
    "height": 300,
    "density": 0.8,
    "maxCircleRadius": 0.1,
    "minCircleRadius": 0.05
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

    def apply_circle(self, circle: Circle) -> None:
        for x in range(max(0, circle.x - circle.radius), min(self.config['width'], circle.x + circle.radius)):
            for y in range(max(0, circle.y - circle.radius), min(self.config['height'], circle.y + circle.radius)):
                distance = math.sqrt((circle.x - x) ** 2 + (circle.y - y) ** 2)
                if distance < circle.radius:
                    self.field[y][x] += 1 - distance / circle.radius

    def place_circles(self):
        min_side = min(self.config['width'], self.config['height'])
        num_circles = min_side * self.config['density']
        min_radius = int(min_side * self.config['minCircleRadius'])
        max_radius = int(min_side * self.config['maxCircleRadius'])

        if self.config["distribution"] == "RANDOM":
            for i in range(int(num_circles)):
                circle = Circle(
                    x=random.randint(0, self.config['width']),
                    y=random.randint(0, self.config['height']),
                    radius=random.randrange(min_radius, max_radius)
                )
                self.apply_circle(circle)
        elif self.config["distribution"] == "EVEN":
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
                    x=int(coord[0]),
                    y=int(coord[1]),
                    radius=random.randrange(min_radius, max_radius)
                )
                self.apply_circle(circle)

                
        else:
            raise Exception(f"Unrecognized distribution: {self.config['distribution']}")
        
    def soften(self):
        self.field = gaussian_filter(self.field, sigma=self.config['blur'])


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

    config = EVEN_CONFIG if len(sys.argv) > 1 and sys.argv[1] == 'EVEN' else RANDOM_CONFIG
    sculptor = Sculptor(config)
    sculptor.generate()
    sculptor.print()
    print("Done!")