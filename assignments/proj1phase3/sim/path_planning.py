from __future__ import annotations

import numpy as np
import heapq


def path_from_a_star(map_points: np.ndarray, grid_size: tuple[int, int, int] = (10, 10, 10)) -> np.ndarray:
    """
    Plan a path on a 3D grid using A* (6-connected).

    Args:
        map_points: An (N, 3) array where:
                    - map_points[0] is the start [x, y, z]
                    - map_points[-1] is the target [x, y, z]
                    - map_points[1:-1] are obstacle locations in the grid.
                    All coordinates are floats but correspond to integer grid centers.
                    Note: The grid indices are 1-based (i.e. x in [1, max_x]).
        grid_size: A tuple (max_x, max_y, max_z) defining the dimensions of the grid.

    Returns:
        np.ndarray: An (M, 3) array of waypoints for the path.
                    The waypoints should be the center of the grid cells (i.e., index - 0.5).
                    The path must start with the start position and end with the target position.
    """
    map_points = np.asarray(map_points, dtype=float)
    if map_points.shape[0] < 2:
        raise ValueError("map_points must contain at least start and target")


    # 1. Parse start, target, and obstacles
    start_idx = tuple(map_points[0].round().astype(int))
    target_idx = tuple(map_points[-1].round().astype(int))
    
    # Create obstacle set for fast lookup
    obstacles = set()
    for obs in map_points[1:-1]:
        obstacles.add(tuple(obs.round().astype(int)))
    
    # 2. Initialize A* data structures
    # Priority queue: (f_score, counter, node)
    # counter is used to break ties
    open_set = []
    counter = 0
    
    # g_score: cost from start to node
    g_score = {start_idx: 0}
    
    # f_score: g_score + heuristic
    def heuristic(a, b):
        # Manhattan distance for 6-connected grid
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])
    
    f_score = {start_idx: heuristic(start_idx, target_idx)}
    
    # Track where we came from for path reconstruction
    came_from = {}
    
    # Add start to open set
    heapq.heappush(open_set, (f_score[start_idx], counter, start_idx))
    counter += 1
    
    # Closed set (visited nodes)
    closed_set = set()
    
    # 6-connected neighbors (±1 in one axis)
    neighbors_delta = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]
    
    # 3. Perform A* search
    while open_set:
        # Get node with lowest f_score
        _, _, current = heapq.heappop(open_set)
        
        # Check if we reached target
        if current == target_idx:
            # 4. Reconstruct path
            path_indices = [current]
            while current in came_from:
                current = came_from[current]
                path_indices.append(current)
            path_indices.reverse()
            
            # Convert to cell-center coordinates (index - 0.5)
            path_points = np.array([[idx[0] - 0.5, idx[1] - 0.5, idx[2] - 0.5] 
                                   for idx in path_indices], dtype=float)
            return path_points
        
        # Skip if already processed
        if current in closed_set:
            continue
        
        closed_set.add(current)
        
        # Explore neighbors
        for dx, dy, dz in neighbors_delta:
            neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
            
            # Check bounds (1-based indexing)
            if not (1 <= neighbor[0] <= grid_size[0] and
                    1 <= neighbor[1] <= grid_size[1] and
                    1 <= neighbor[2] <= grid_size[2]):
                continue
            
            # Check if obstacle
            if neighbor in obstacles:
                continue
            
            # Skip if already processed
            if neighbor in closed_set:
                continue
            
            # Calculate tentative g_score
            tentative_g = g_score[current] + 1  # Each step costs 1
            
            # If this path is better, update
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, target_idx)
                heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
                counter += 1

    raise NotImplementedError("path_from_a_star not implemented")
