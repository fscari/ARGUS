class GridCache:
    def __init__(self, y_threshold=10):
        self.grid = None
        self.road_mask = None
        self.road_cells = None
        self.last_update_frame = -1
        self.prev_y_min = None
        self.prev_y_max = None
        self.y_threshold = y_threshold

    def needs_update(self, y_min, y_max):
        if self.grid is None or self.road_cells is None:
            return True

        # Check if the y_min or y_max has changed significantly
        if self.prev_y_min is None or self.prev_y_max is None:
            self.prev_y_min = y_min
            self.prev_y_max = y_max
            return True

        y_min_change = y_min - self.prev_y_min
        y_max_change = y_max - self.prev_y_max
        # print(y_min_change)
        if y_min_change < 0:
            self.prev_y_min = y_min
            return True
        elif y_max_change > 0:
            self.prev_y_max = y_max
            return True
        return False

    def update_cache(self, new_grid, new_road_cells):
        # self.grid = new_grid
        # self.road_mask = set(new_grid.keys())
        # self.road_cells = new_road_cells
        if self.grid is None:
            # If the grid is not yet initialized, set it directly
            self.grid = new_grid
        else:
            # Merge the existing grid with the new grid
            for key, value in new_grid.items():
                if key in self.grid:
                    # If the cell already exists, append new points to the existing points
                    self.grid[key].extend(value)
                else:
                    # If the cell doesn't exist, add it to the grid
                    self.grid[key] = value

            # Update the road_mask with the new grid keys
        self.road_mask = set(self.grid.keys())

        # Merge road cells similarly if needed or update directly
        if self.road_cells is None:
            self.road_cells = new_road_cells
        else:
            self.road_cells.update(new_road_cells)

        # Calculate the extents
        grid_x_values = [key[0] for key in  self.grid.keys()]
        grid_y_values = [key[1] for key in  self.grid.keys()]

        x_min = min(grid_x_values)
        x_max = max(grid_x_values)
        y_min = min(grid_y_values)
        y_max = max(grid_y_values)

        # Check if the grid covers the required area
        expected_y_min = -78
        expected_y_max = 78
        expected_x_max = 9

        if y_min <= expected_y_min and y_max >= expected_y_max:
            print("The y-range of the grid is sufficient.")
        else:
            print(f"Warning: The y-range is insufficient. y_min = {y_min}, y_max = {y_max}, expected y_min = {expected_y_min}, expected y_max = {expected_y_max}.")

        if x_max >= expected_x_max:
            print("The x-range of the grid is sufficient.")
        else:
            print(f"Warning: The x-range is insufficient. x_max = {x_max}, expected x_max = {expected_x_max}.")
