import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from genetic_v2 import TestCurve

# Target broken line
target_points = [(0, 0), (50, 50), (100, 0), (150, 50), (200, 0)]  
target_line = TestCurve.create_broken_line(target_points)

# Random drone path (simulating a less fit individual)
random_points = [(0, 0), (50, 50), (100, 0), (150, 50), (190, 0)]  
# random_points = [(0, 10), (40, 60), (80, 10), (160, 60), (200, 10)]  
random_line = TestCurve.create_broken_line(random_points)

# Create Shapely polygons
target_polygon = Polygon(target_points)
random_polygon = Polygon(random_points)

# Calculate area difference
intersection_area = target_polygon.intersection(random_polygon).area
union_area = target_polygon.union(random_polygon).area
area_similarity = intersection_area / union_area
area_difference = 1 - area_similarity  # Inverted for consistency with fitness function

# Plotting
plt.figure(figsize=(10, 6))

# Plot the target broken line
plt.plot(target_line[:, 0], target_line[:, 1], 'b-', label='Target Line', linewidth=2)
plt.fill(target_line[:, 0], target_line[:, 1], 'lightblue', alpha=0.5)  # Fill the target area

# Plot the random broken line
plt.plot(random_line[:, 0], random_line[:, 1], 'r-', label='Random Drone Path', linewidth=2)
plt.fill(random_line[:, 0], random_line[:, 1], 'lightcoral', alpha=0.5)  # Fill the random area

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Broken Line Paths (Area Difference: {area_difference:.4f})')
plt.legend()

plt.grid(alpha=0.4)
plt.savefig(f"area.png")
plt.show()