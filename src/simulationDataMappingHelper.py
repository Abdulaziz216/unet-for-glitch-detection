# Filename: simulatedDataMappingHelper.py
# Author: SUDS2025
# Created: 2024-08-04
# Description: Contains helper functions for simulation data mapping.

# *Mapping from timeseries to movie*

import numpy as np
from scipy.sparse import coo_matrix

# useful units
deg = np.deg2rad(1)
arcmin = deg / 60
s = 1 # second
Hz = 1/s

class FocalPlaneGridBilinear:
    """this works regular spaced detectors on a grid such as in this case, but not
    with actual data which don't fall on a grid"""
    def __init__(self, fplane, pixel_size=0.1*arcmin):
        self.fplane = fplane
        self.pixel_size = pixel_size
        # The create_grid method now returns the 1D coordinate arrays as well for clarity
        self.grid, self.grid_shape, self.grid_coords_1d = self._create_grid()
        self.weight_matrix = self._precompute_weights()
        # You can check if the matrix was created successfully
        print(f"Weight matrix created with {self.weight_matrix.nnz} non-zero elements.")

    def _create_grid(self):
        """Create regular grid covering focal plane"""
        pad = 0.1 * (np.max(self.fplane.x) - np.min(self.fplane.x))
        x_min, x_max = np.min(self.fplane.x) - pad, np.max(self.fplane.x) + pad
        y_min, y_max = np.min(self.fplane.y) - pad, np.max(self.fplane.y) + pad

        nx = int(np.ceil((x_max - x_min) / self.pixel_size))
        ny = int(np.ceil((y_max - y_min) / self.pixel_size))

        grid_x_1d = np.linspace(x_min, x_max, nx)
        grid_y_1d = np.linspace(y_min, y_max, ny)

        # np.meshgrid with 'xy' indexing returns X, Y
        grid_x, grid_y = np.meshgrid(grid_x_1d, grid_y_1d)

        return (grid_x, grid_y), (ny, nx), (grid_x_1d, grid_y_1d)

    def _precompute_weights(self):
        """Precompute bilinear interpolation weights for all grid points"""
        grid_x_1d, grid_y_1d = self.grid_coords_1d
        grid_x_2d, grid_y_2d = self.grid
        detector_points = np.column_stack((self.fplane.x, self.fplane.y))

        rows, cols, weights = [], [], []

        for det_idx, (x_det, y_det) in enumerate(detector_points):
            # Find grid cell containing detector using the correct 1D coordinate arrays
            i = np.searchsorted(grid_x_1d, x_det) - 1
            j = np.searchsorted(grid_y_1d, y_det) - 1

            # Skip if detector outside grid
            if i < 0 or i >= self.grid_shape[1]-1 or j < 0 or j >= self.grid_shape[0]-1:
                continue

            # Get cell corners from the 2D grid arrays
            x0, x1 = grid_x_2d[j, i], grid_x_2d[j, i+1]
            y0, y1 = grid_y_2d[j, i], grid_y_2d[j+1, i]

            # Bilinear weights calculation
            dx = (x_det - x0) / (x1 - x0) if x1 != x0 else 0
            dy = (y_det - y0) / (y1 - y0) if y1 != y0 else 0

            # Corner indices (row-major order)
            idx00 = j * self.grid_shape[1] + i
            idx01 = j * self.grid_shape[1] + i + 1
            idx10 = (j + 1) * self.grid_shape[1] + i
            idx11 = (j + 1) * self.grid_shape[1] + i + 1

            # Weight calculation
            w00 = (1 - dx) * (1 - dy)
            w01 = dx * (1 - dy)
            w10 = (1 - dx) * dy
            w11 = dx * dy

            if np.isnan([w00, w01, w10, w11]).any():
                continue

            # Store weights
            rows.extend([idx00, idx01, idx10, idx11])
            cols.extend([det_idx] * 4)
            weights.extend([w00, w01, w10, w11])

        if not weights:
            print("Warning: No detectors were found within the grid boundaries. The weight matrix will be empty.")

        return coo_matrix((weights, (rows, cols)),
                          shape=(self.grid_shape[0]*self.grid_shape[1],
                                 len(self.fplane.x)))

    def tod_to_video(self, tod_data):
        """Convert TOD data to video using precomputed weights"""
        if tod_data.ndim == 1:
            tod_data = tod_data.reshape(1, -1)
        if tod_data.shape[0] == len(self.fplane.x):
            tod_data = tod_data.T

        if tod_data.shape[1] != len(self.fplane.x):
            raise ValueError("Shape of tod_data does not match number of detectors")

        video_flat = self.weight_matrix.dot(tod_data.T).T
        return video_flat.reshape(-1, self.grid_shape[0], self.grid_shape[1])

import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
from scipy.spatial import Delaunay


class FocalPlaneFlex:
    def __init__(self, fplane, grid_resolution=50):
        self.fplane = fplane
        self.grid_resolution = grid_resolution

        # Create detector coordinate array for triangulation
        self.detector_coords = np.column_stack((self.fplane.x, self.fplane.y))

        # Create triangulation from detector positions
        self.triangulation = Delaunay(self.detector_coords)

        # Create regular output grid and precompute interpolation weights
        self.output_grid, self.grid_shape = self._create_output_grid()
        self.weight_matrix, self.valid_mask = self._precompute_weights()

        print(f"Triangulation created with {len(self.triangulation.simplices)} triangles")
        print(f"Weight matrix created with {self.weight_matrix.nnz} non-zero elements")
        print(f"Valid grid points: {np.sum(self.valid_mask)} / {len(self.valid_mask)}")

    def _create_output_grid(self):
        """Create regular output grid for visualization"""
        # Add padding around detector positions
        pad = 0.1 * (np.max(self.fplane.x) - np.min(self.fplane.x))
        x_min, x_max = np.min(self.fplane.x) - pad, np.max(self.fplane.x) + pad
        y_min, y_max = np.min(self.fplane.y) - pad, np.max(self.fplane.y) + pad

        # Create regular grid for output
        x_grid = np.linspace(x_min, x_max, self.grid_resolution)
        y_grid = np.linspace(y_min, y_max, self.grid_resolution)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Flatten for easier processing
        grid_points = np.column_stack((X.flatten(), Y.flatten()))

        return grid_points, (self.grid_resolution, self.grid_resolution)

    def _precompute_weights(self):
        """Precompute barycentric interpolation weights using Delaunay triangulation"""
        n_grid_points = len(self.output_grid)
        n_detectors = len(self.detector_coords)

        rows, cols, weights = [], [], []
        valid_mask = np.zeros(n_grid_points, dtype=bool)

        # Find which triangle each grid point belongs to
        simplex_indices = self.triangulation.find_simplex(self.output_grid)

        for grid_idx, simplex_idx in enumerate(simplex_indices):
            if simplex_idx == -1:
                # Point is outside convex hull
                continue

            valid_mask[grid_idx] = True

            # Get the vertices of the triangle containing this grid point
            triangle_vertices = self.triangulation.simplices[simplex_idx]

            # Calculate barycentric coordinates
            barycentric_coords = self._calculate_barycentric_coordinates(
                self.output_grid[grid_idx],
                self.detector_coords[triangle_vertices]
            )

            # Store weights for each vertex of the triangle
            for i, vertex_idx in enumerate(triangle_vertices):
                weight = barycentric_coords[i]
                if weight > 1e-10:  # Only store significant weights
                    rows.append(grid_idx)
                    cols.append(vertex_idx)
                    weights.append(weight)

        weight_matrix = coo_matrix(
            (weights, (rows, cols)),
            shape=(n_grid_points, n_detectors)
        )

        return weight_matrix, valid_mask

    def _calculate_barycentric_coordinates(self, point, triangle_vertices):
        """Calculate barycentric coordinates of a point within a triangle"""
        # Triangle vertices
        A, B, C = triangle_vertices

        # Vectors
        v0 = C - A
        v1 = B - A
        v2 = point - A

        # Compute dot products
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        # Compute barycentric coordinates
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        w = 1 - u - v

        return np.array([w, v, u])  # Corresponds to vertices A, B, C

    def tod_to_video(self, tod_data):
        """Convert TOD data to video using precomputed triangulation weights"""
        if tod_data.ndim == 1:
            tod_data = tod_data.reshape(1, -1)
        if tod_data.shape[0] == len(self.fplane.x):
            tod_data = tod_data.T

        if tod_data.shape[1] != len(self.fplane.x):
            raise ValueError(f"Shape of tod_data {tod_data.shape} does not match number of detectors {len(self.fplane.x)}")

        # Interpolate data to grid points
        video_flat = self.weight_matrix.dot(tod_data.T).T

        # Set invalid (extrapolated) points to zero
        video_flat[:, ~self.valid_mask] = 0

        # Reshape to video format
        n_frames = video_flat.shape[0]
        video = video_flat.reshape(n_frames, self.grid_shape[0], self.grid_shape[1])

        return video

    def get_detector_triangulation_info(self):
        """Return information about the triangulation for debugging/visualization"""
        return {
            'triangles': self.triangulation.simplices,
            'detector_coords': self.detector_coords,
            'n_triangles': len(self.triangulation.simplices),
            'convex_hull': self.triangulation.convex_hull
        }

    def plot_triangulation(self):
        """Plot the triangulation for visualization (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Plot triangulation
            ax1.triplot(self.detector_coords[:, 0], self.detector_coords[:, 1],
                       self.triangulation.simplices, 'k-', alpha=0.3)
            ax1.plot(self.detector_coords[:, 0], self.detector_coords[:, 1], 'ro', markersize=3)
            ax1.set_title('Delaunay Triangulation of Detectors')
            ax1.set_xlabel('X coordinate')
            ax1.set_ylabel('Y coordinate')
            ax1.grid(True, alpha=0.3)

            # Plot output grid with valid points highlightedc
            grid_2d = self.output_grid.reshape(self.grid_shape[0], self.grid_shape[1], 2)
            valid_2d = self.valid_mask.reshape(self.grid_shape[0], self.grid_shape[1])

            ax2.contourf(grid_2d[:, :, 0], grid_2d[:, :, 1], valid_2d.astype(int),
                        levels=[0, 0.5, 1], colors=['lightgray', 'lightblue'], alpha=0.7)
            ax2.plot(self.detector_coords[:, 0], self.detector_coords[:, 1], 'ro', markersize=3)
            ax2.set_title('Output Grid (Blue=Valid, Gray=Extrapolated)')
            ax2.set_xlabel('X coordinate')
            ax2.set_ylabel('Y coordinate')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Matplotlib not available for plotting")










import numpy as np
from scipy.sparse import coo_matrix


# class FocalPlaneGridBilinear:
#     """this works regular spaced detectors on a grid such as in this case, but not
#     with actual data which don't fall on a grid"""
#     def __init__(self, fplane, pixel_size=0.1*arcmin):
#         self.fplane = fplane
#         self.pixel_size = pixel_size
#         # The create_grid method now returns the 1D coordinate arrays as well for clarity
#         self.grid, self.grid_shape, self.grid_coords_1d = self._create_grid()
#         self.weight_matrix = self._precompute_weights()
#         # You can check if the matrix was created successfully
#         print(f"Weight matrix created with {self.weight_matrix.nnz} non-zero elements.")

#     def _create_grid(self):
#         """Create regular grid covering focal plane"""
#         pad = 0.1 * (np.max(self.fplane.x) - np.min(self.fplane.x))
#         x_min, x_max = np.min(self.fplane.x) - pad, np.max(self.fplane.x) + pad
#         y_min, y_max = np.min(self.fplane.y) - pad, np.max(self.fplane.y) + pad

#         nx = int(np.ceil((x_max - x_min) / self.pixel_size))
#         ny = int(np.ceil((y_max - y_min) / self.pixel_size))

#         grid_x_1d = np.linspace(x_min, x_max, nx)
#         grid_y_1d = np.linspace(y_min, y_max, ny)

#         # np.meshgrid with 'xy' indexing returns X, Y
#         grid_x, grid_y = np.meshgrid(grid_x_1d, grid_y_1d)

#         return (grid_x, grid_y), (ny, nx), (grid_x_1d, grid_y_1d)

#     def _precompute_weights(self):
#         """Precompute bilinear interpolation weights for all grid points"""
#         grid_x_1d, grid_y_1d = self.grid_coords_1d
#         grid_x_2d, grid_y_2d = self.grid
#         detector_points = np.column_stack((self.fplane.x, self.fplane.y))

#         rows, cols, weights = [], [], []

#         for det_idx, (x_det, y_det) in enumerate(detector_points):
#             # Find grid cell containing detector using the correct 1D coordinate arrays
#             i = np.searchsorted(grid_x_1d, x_det) - 1
#             j = np.searchsorted(grid_y_1d, y_det) - 1

#             # Skip if detector outside grid
#             if i < 0 or i >= self.grid_shape[1]-1 or j < 0 or j >= self.grid_shape[0]-1:
#                 continue

#             # Get cell corners from the 2D grid arrays
#             x0, x1 = grid_x_2d[j, i], grid_x_2d[j, i+1]
#             y0, y1 = grid_y_2d[j, i], grid_y_2d[j+1, i]

#             # Bilinear weights calculation
#             dx = (x_det - x0) / (x1 - x0) if x1 != x0 else 0
#             dy = (y_det - y0) / (y1 - y0) if y1 != y0 else 0

#             # Corner indices (row-major order)
#             idx00 = j * self.grid_shape[1] + i
#             idx01 = j * self.grid_shape[1] + i + 1
#             idx10 = (j + 1) * self.grid_shape[1] + i
#             idx11 = (j + 1) * self.grid_shape[1] + i + 1

#             # Weight calculation
#             w00 = (1 - dx) * (1 - dy)
#             w01 = dx * (1 - dy)
#             w10 = (1 - dx) * dy
#             w11 = dx * dy

#             if np.isnan([w00, w01, w10, w11]).any():
#                 continue

#             # Store weights
#             rows.extend([idx00, idx01, idx10, idx11])
#             cols.extend([det_idx] * 4)
#             weights.extend([w00, w01, w10, w11])

#         if not weights:
#             print("Warning: No detectors were found within the grid boundaries. The weight matrix will be empty.")

#         return coo_matrix((weights, (rows, cols)),
#                           shape=(self.grid_shape[0]*self.grid_shape[1],
#                                  len(self.fplane.x)))

#     def tod_to_video(self, tod_data):
#         """Convert TOD data to video using precomputed weights"""
#         if tod_data.ndim == 1:
#             tod_data = tod_data.reshape(1, -1)
#         if tod_data.shape[0] == len(self.fplane.x):
#             tod_data = tod_data.T

#         if tod_data.shape[1] != len(self.fplane.x):
#             raise ValueError("Shape of tod_data does not match number of detectors")

#         video_flat = self.weight_matrix.dot(tod_data.T).T
#         return video_flat.reshape(-1, self.grid_shape[0], self.grid_shape[1])

import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
from scipy.spatial import Delaunay
from scipy.interpolate import RegularGridInterpolator

class FocalPlaneFlex:
    def __init__(self, fplane, grid_resolution=50):
        self.fplane = fplane
        self.grid_resolution = grid_resolution

        # Create detector coordinate array for triangulation
        self.detector_coords = np.column_stack((self.fplane.x, self.fplane.y))

        # Create triangulation from detector positions
        self.triangulation = Delaunay(self.detector_coords)

        # Create regular output grid and precompute interpolation weights
        self.output_grid, self.x_grid, self.y_grid, self.grid_shape = self._create_output_grid()
        self.weight_matrix, self.valid_mask = self._precompute_weights()

        print(f"Triangulation created with {len(self.triangulation.simplices)} triangles")
        print(f"Weight matrix created with {self.weight_matrix.nnz} non-zero elements")
        print(f"Valid grid points: {np.sum(self.valid_mask)} / {len(self.valid_mask)}")

    def _create_output_grid(self):
        """Create regular output grid for visualization"""
        pad = 0.1 * (np.max(self.fplane.x) - np.min(self.fplane.x))
        x_min, x_max = np.min(self.fplane.x) - pad, np.max(self.fplane.x) + pad
        y_min, y_max = np.min(self.fplane.y) - pad, np.max(self.fplane.y) + pad

        # Create regular grid for output
        x_grid = np.linspace(x_min, x_max, self.grid_resolution)
        y_grid = np.linspace(y_min, y_max, self.grid_resolution)
        X, Y = np.meshgrid(x_grid, y_grid)

        grid_points = np.column_stack((X.flatten(), Y.flatten()))

        # Return the grid axes as well
        return grid_points, x_grid, y_grid, (self.grid_resolution, self.grid_resolution)


    def _precompute_weights(self):
        """Precompute barycentric interpolation weights using Delaunay triangulation"""
        n_grid_points = len(self.output_grid)
        n_detectors = len(self.detector_coords)

        rows, cols, weights = [], [], []
        valid_mask = np.zeros(n_grid_points, dtype=bool)

        # Find which triangle each grid point belongs to
        simplex_indices = self.triangulation.find_simplex(self.output_grid)

        for grid_idx, simplex_idx in enumerate(simplex_indices):
            if simplex_idx == -1:
                # Point is outside convex hull
                continue

            valid_mask[grid_idx] = True

            # Get the vertices of the triangle containing this grid point
            triangle_vertices = self.triangulation.simplices[simplex_idx]

            # Calculate barycentric coordinates
            barycentric_coords = self._calculate_barycentric_coordinates(
                self.output_grid[grid_idx],
                self.detector_coords[triangle_vertices]
            )

            # Store weights for each vertex of the triangle
            for i, vertex_idx in enumerate(triangle_vertices):
                weight = barycentric_coords[i]
                if weight > 1e-10:  # Only store significant weights
                    rows.append(grid_idx)
                    cols.append(vertex_idx)
                    weights.append(weight)

        weight_matrix = coo_matrix(
            (weights, (rows, cols)),
            shape=(n_grid_points, n_detectors)
        )

        return weight_matrix, valid_mask

    def _calculate_barycentric_coordinates(self, point, triangle_vertices):
        """Calculate barycentric coordinates of a point within a triangle"""
        # Triangle vertices
        A, B, C = triangle_vertices

        # Vectors
        v0 = C - A
        v1 = B - A
        v2 = point - A

        # Compute dot products
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        # Compute barycentric coordinates
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        w = 1 - u - v

        return np.array([w, v, u])  # Corresponds to vertices A, B, C

    def tod_to_video(self, tod_data):
        """Convert TOD data to video using precomputed triangulation weights"""
        if tod_data.ndim == 1:
            tod_data = tod_data.reshape(1, -1)
        if tod_data.shape[0] == len(self.fplane.x):
            tod_data = tod_data.T

        if tod_data.shape[1] != len(self.fplane.x):
            raise ValueError(f"Shape of tod_data {tod_data.shape} does not match number of detectors {len(self.fplane.x)}")

        # Interpolate data to grid points
        video_flat = self.weight_matrix.dot(tod_data.T).T

        # Set invalid (extrapolated) points to zero
        video_flat[:, ~self.valid_mask] = 0

        # Reshape to video format
        n_frames = video_flat.shape[0]
        video = video_flat.reshape(n_frames, self.grid_shape[0], self.grid_shape[1])

        return video

    def video_to_tod(self, video_data):
        """
        Maps image or video data back to the original detector positions.
        This is the reverse operation of tod_to_video.

        Args:
            video_data (np.ndarray): A single image (H, W) or a sequence of
                                    images (n_frames, H, W). The dimensions
                                    H and W must match self.grid_resolution.

        Returns:
            np.ndarray: The resampled detector data (TOD) with shape
                        (n_frames, n_detectors).
        """
        if video_data.ndim == 2:
            video_data = video_data[np.newaxis, :, :]

        n_frames, h, w = video_data.shape
        if (h, w) != self.grid_shape:
            raise ValueError(
                f"Input video shape {(h, w)} does not match grid shape {self.grid_shape}"
            )

        query_points = self.detector_coords
        n_detectors = len(query_points)

        tod_data = np.zeros((n_frames, n_detectors))

        for i in range(n_frames):
            frame = video_data[i]

            interpolator = RegularGridInterpolator(
                (self.y_grid, self.x_grid),
                frame,
                method='linear',
                bounds_error=False,
                fill_value=0
            )
            tod_data[i, :] = interpolator(query_points[:, ::-1])

        if tod_data.shape[0] == 1:
            return tod_data.flatten()

        return tod_data

    def get_detector_triangulation_info(self):
        """Return information about the triangulation for debugging/visualization"""
        return {
            'triangles': self.triangulation.simplices,
            'detector_coords': self.detector_coords,
            'n_triangles': len(self.triangulation.simplices),
            'convex_hull': self.triangulation.convex_hull
        }

    def plot_triangulation(self):
        """Plot the triangulation for visualization (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Plot triangulation
            ax1.triplot(self.detector_coords[:, 0], self.detector_coords[:, 1],
                       self.triangulation.simplices, 'k-', alpha=0.3)
            ax1.plot(self.detector_coords[:, 0], self.detector_coords[:, 1], 'ro', markersize=3)
            ax1.set_title('Delaunay Triangulation of Detectors')
            ax1.set_xlabel('X coordinate')
            ax1.set_ylabel('Y coordinate')
            ax1.grid(True, alpha=0.3)

            # Plot output grid with valid points highlightedc
            grid_2d = self.output_grid.reshape(self.grid_shape[0], self.grid_shape[1], 2)
            valid_2d = self.valid_mask.reshape(self.grid_shape[0], self.grid_shape[1])

            ax2.contourf(grid_2d[:, :, 0], grid_2d[:, :, 1], valid_2d.astype(int),
                        levels=[0, 0.5, 1], colors=['lightgray', 'lightblue'], alpha=0.7)
            ax2.plot(self.detector_coords[:, 0], self.detector_coords[:, 1], 'ro', markersize=3)
            ax2.set_title('Output Grid (Blue=Valid, Gray=Extrapolated)')
            ax2.set_xlabel('X coordinate')
            ax2.set_ylabel('Y coordinate')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Matplotlib not available for plotting")
