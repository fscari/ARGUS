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
        self.grid = new_grid
        self.road_mask = set(new_grid.keys())
        self.road_cells = new_road_cells
