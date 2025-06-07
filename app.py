import plotly.graph_objects as go
from io import BytesIO
import streamlit as st
from stl import mesh
import numpy as np
import time


def ray_intersects_triangle(ray_origin, ray_direction, vertex0, vertex1, vertex2):
    eps = 1e-8

    edge1 = vertex1 - vertex0
    edge2 = vertex2 - vertex0

    ray_cross = np.cross(ray_direction, edge2)
    det = np.dot(edge1, ray_cross)

    if -eps < det < eps:
        return False, None

    det_inv = 1.0 / det
    ray_to_vertex = ray_origin - vertex0

    u_bar_coord = det_inv * np.dot(ray_to_vertex, ray_cross)

    if u_bar_coord < 0.0 or u_bar_coord > 1.0:
        return False, None

    ray_to_vertex_cross = np.cross(ray_to_vertex, edge1)
    v_bar_coord = det_inv * np.dot(ray_direction, ray_to_vertex_cross)
    if v_bar_coord < 0.0 or u_bar_coord + v_bar_coord > 1.0:
        return False, None

    t_distance = det_inv * np.dot(edge2, ray_to_vertex_cross)
    if t_distance > eps:
        intersection = ray_origin + ray_direction * t_distance
        return True, intersection[2]
    else:
        return False, None


class Solid:
    def __init__(self, stl_mesh=None):
        self.stl_mesh = stl_mesh
        self.vertices = None
        self.faces = None
        self.triangle_bboxes = None

        if stl_mesh is not None:
            self.process_stl()

    def process_stl(self):
        self.vertices = self.stl_mesh.vectors.reshape(-1, 3)
        self.faces = np.arange(len(self.vertices)).reshape(-1, 3)

        tris = self.vertices[self.faces]
        self.triangle_bboxes = np.empty((len(self.faces), 2, 3))
        self.triangle_bboxes[:, 0, :] = np.min(tris, axis=1)
        self.triangle_bboxes[:, 1, :] = np.max(tris, axis=1)

    def is_point_inside(self, point):
        ray_direction_vector = np.array([1, 0, 0])
        intersection_count = 0

        y_in_range = (point[1] >= self.triangle_bboxes[:, 0, 1]) & (
            point[1] <= self.triangle_bboxes[:, 1, 1]
        )
        z_in_range = (point[2] >= self.triangle_bboxes[:, 0, 2]) & (
            point[2] <= self.triangle_bboxes[:, 1, 2]
        )
        candidates = np.where(y_in_range & z_in_range)[0]

        for i in candidates:
            vertex0 = self.vertices[self.faces[i][0]]
            vertex1 = self.vertices[self.faces[i][1]]
            vertex2 = self.vertices[self.faces[i][2]]

            if ray_intersects_triangle(
                point, ray_direction_vector, vertex0, vertex1, vertex2
            )[0]:
                intersection_count += 1

        return intersection_count % 2 == 1

    def contains(self, points):
        return np.array([self.is_point_inside(point) for point in points], dtype=bool)

    def info(self):
        if self.stl_mesh is None:
            return "Nie wczytano pliku STL."
        else:
            return {
                "Liczba wierzchołków": len(self.vertices),
                "Liczba ścian": len(self.faces),
                "Wymiary": {
                    "X": [
                        float(np.min(self.vertices[:, 0])),
                        float(np.max(self.vertices[:, 0])),
                    ],
                    "Y": [
                        float(np.min(self.vertices[:, 1])),
                        float(np.max(self.vertices[:, 1])),
                    ],
                    "Z": [
                        float(np.min(self.vertices[:, 2])),
                        float(np.max(self.vertices[:, 2])),
                    ],
                },
            }


class MonteCarloVolumeEstimator:
    def __init__(self, solid, num_points=10000):
        self.solid = solid
        self.num_points = num_points
        self.points = None
        self.inside_points = None
        self.bounding_box = self.compute_bounding_box()

    def compute_bounding_box(self):
        vertices = self.solid.vertices
        min_bounds = np.min(vertices, axis=0)
        max_bounds = np.max(vertices, axis=0)

        return min_bounds, max_bounds

    def generate_random_points(self):
        min_bounds, max_bounds = self.bounding_box

        self.points = np.random.uniform(
            low=min_bounds,
            high=max_bounds,
            size=(self.num_points, 3),
        )

    def get_points(self):
        return self.points, self.inside_points

    def run(self, progress_callback=None):
        self.generate_random_points()

        total = len(self.points)
        inside = np.zeros(total, dtype=bool)

        for i, point in enumerate(self.points):
            inside[i] = self.solid.is_point_inside(point)

            if progress_callback and i % max(1, total // 100) == 0:
                progress_callback(i / total)

        self.inside_points = inside

        bounding_box_volume = np.prod(self.bounding_box[1] - self.bounding_box[0])
        volume_estimate = (
            np.sum(self.inside_points) / self.num_points
        ) * bounding_box_volume

        return volume_estimate


class CuboidVolumeEstimator:
    def __init__(self, solid, cuboid_size_x=1.0, cuboid_size_y=1.0):
        self.solid = solid
        self.cuboid_size_x = cuboid_size_x
        self.cuboid_size_y = cuboid_size_y
        self.cube_centers = []
        self.heights = []

    def run(self, progress_callback=None):
        min_bounds, max_bounds = self.solid.vertices.min(
            axis=0
        ), self.solid.vertices.max(axis=0)

        x_start, x_end = min_bounds[0], max_bounds[0]
        y_start, y_end = min_bounds[1], max_bounds[1]

        nx = int((x_end - x_start) // self.cuboid_size_x)
        ny = int((y_end - y_start) // self.cuboid_size_y)

        volume = 0.0
        self.cube_centers = []
        self.heights = []

        for i in range(nx):
            for j in range(ny):
                if progress_callback and (i * ny + j) % max(1, nx * ny // 100) == 0:
                    progress_callback((i * ny + j) / (nx * ny))

                x = x_start + (i + 0.5) * self.cuboid_size_x
                y = y_start + (j + 0.5) * self.cuboid_size_y

                z_max = self.find_top_surface_height(np.array([x, y, min_bounds[2]]))

                if z_max is not None:
                    h = z_max - min_bounds[2]
                    v = self.cuboid_size_x * self.cuboid_size_y * h
                    volume += v

                    self.cube_centers.append([x, y, min_bounds[2] + h / 2.0])
                    self.heights.append(h)

        return volume

    def find_top_surface_height(self, origin):
        ray_direction = np.array([0, 0, 1])
        max_z = None

        y_in_range = (origin[1] >= self.solid.triangle_bboxes[:, 0, 1]) & (
            origin[1] <= self.solid.triangle_bboxes[:, 1, 1]
        )
        x_in_range = (origin[0] >= self.solid.triangle_bboxes[:, 0, 0]) & (
            origin[0] <= self.solid.triangle_bboxes[:, 1, 0]
        )
        candidates = np.where(y_in_range & x_in_range)[0]

        for idx in candidates:
            vertex0 = self.solid.vertices[self.solid.faces[idx][0]]
            vertex1 = self.solid.vertices[self.solid.faces[idx][1]]
            vertex2 = self.solid.vertices[self.solid.faces[idx][2]]

            intersect, z_val = ray_intersects_triangle(
                origin, ray_direction, vertex0, vertex1, vertex2
            )
            if intersect:
                if max_z is None or z_val > max_z:
                    max_z = z_val

        return max_z


class App:
    def __init__(self):
        self.solid = None
        self.plot_container = None
        self.default_plot = None
        self.cube_size = None

    def load_stl(self):
        st.sidebar.title("Wczytaj plik STL")
        uploaded_file = st.sidebar.file_uploader("Wybierz plik STL", type=["stl"])

        if uploaded_file is not None:
            stl_mesh = mesh.Mesh.from_file(
                uploaded_file.name, fh=BytesIO(uploaded_file.read())
            )
            self.solid = Solid(stl_mesh)
        else:
            st.info("Proszę wczytać plik STL.")

    def display_stl(self):
        if self.solid is None:
            st.warning("Nie wczytano pliku STL.")
            return

        if self.plot_container is not None:
            self.plot_container.empty()

        fig = go.Figure()

        vertices = self.solid.vertices
        faces = self.solid.faces

        fig.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=0.5,
                color="lightblue",
            )
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data",
            ),
            title="Bryła",
            margin=dict(l=0, r=0, b=0, t=30),
            height=500,
        )

        if self.plot_container is None:
            self.plot_container = st.empty()
        else:
            self.plot_container.empty()

        self.plot_container.plotly_chart(fig, use_container_width=True, key="plot")
        self.default_plot = fig

    def display_info(self):
        if self.solid is None:
            return

        st.sidebar.divider()
        st.sidebar.subheader("Informacje o modelu")

        info = self.solid.info()

        for key, value in info.items():
            if isinstance(value, dict):
                st.sidebar.write(f"{key}:")
                for sub_key, sub_value in value.items():
                    st.sidebar.write(f"  {sub_key}: {sub_value}")
                    pass
            else:
                st.sidebar.write(f"{key}: {value}")

    def monte_carlo_ui(self):
        if self.solid is None:
            return

        st.sidebar.subheader("metoda Monte Carlo")

        num_points = st.sidebar.number_input(
            "Liczba punktów Monte Carlo",
            min_value=10,
            max_value=1000000,
            value=1000,
            step=100,
            format="%d",
            help="Liczba punktów Monte Carlo do oszacowania objętości.",
        )

        calculate_button = st.sidebar.button(
            "Oblicz objętość",
            key="calculate_button",
            args=(True,),
        )

        if calculate_button:
            if self.plot_container is not None:
                self.plot_container.empty()

            def update_progress(pct):
                progress_bar.progress(
                    min(int(pct * 100), 100), text=f"Obliczanie... {int(pct * 100)}%"
                )

            start_time = time.time()
            estimator = MonteCarloVolumeEstimator(self.solid, num_points)

            progress_bar = st.progress(0, text="Obliczanie objętości...")

            volume = estimator.run(progress_callback=update_progress)
            #
            points, inside_points = estimator.get_points()
            self.plot_monte_carlo(points, inside_points)

            elapsed_time = time.time() - start_time
            st.sidebar.success(f"Przybliżona objętość: {volume:.2f} jednostek^3")
            st.sidebar.info(f"Czas obliczeń: {elapsed_time:.2f} sekund")
            progress_bar.empty()

    def cube_ui(self):
        if self.solid is None:
            st.warning("Nie wczytano pliku STL.")
            return

        st.sidebar.subheader("metoda prostokątów")

        cube_size_a = st.sidebar.number_input(
            "Długość prostopadłościanu",
            min_value=0.01,
            max_value=100.0,
            value=1.0,
            step=0.01,
            format="%.2f",
            help="Rozmiar podstawy sześcianu.",
        )

        cube_size_b = st.sidebar.number_input(
            "Szerokość prostopadłościanu",
            min_value=0.01,
            max_value=100.0,
            value=1.0,
            step=0.01,
            format="%.2f",
            help="Rozmiar podstawy sześcianu.",
        )

        calculate_button = st.sidebar.button("Oblicz objętość", key="calculate_button2")

        if calculate_button:
            if self.plot_container is not None:
                self.plot_container.empty()

            def update_progress(pct):
                progress_bar.progress(
                    min(int(pct * 100), 100), text=f"Obliczanie... {int(pct * 100)}%"
                )

            start_time = time.time()
            estimator = CuboidVolumeEstimator(
                self.solid, cuboid_size_x=cube_size_a, cuboid_size_y=cube_size_b
            )

            progress_bar = st.progress(0, text="Obliczanie objętości...")

            volume = estimator.run(progress_callback=update_progress)
            cube_centers, heights = estimator.cube_centers, estimator.heights
            self.cube_size = (cube_size_a, cube_size_b)
            self.plot_cuboids(cube_centers, heights)

            elapsed_time = time.time() - start_time
            st.sidebar.success(f"Przybliżona objętość: {volume:.2f} jednostek^3")
            st.sidebar.info(f"Czas obliczeń: {elapsed_time:.2f} sekund")

            progress_bar.empty()

    def plot_monte_carlo(self, points, inside_points):
        fig = go.Figure()

        vertices = self.solid.vertices
        faces = self.solid.faces

        fig.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=0.25,
                color="lightblue",
                name="Bryła",
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=points[inside_points, 0],
                y=points[inside_points, 1],
                z=points[inside_points, 2],
                mode="markers",
                marker=dict(size=2, color="green"),
                name="Punkty wewnętrzne",
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=points[~inside_points, 0],
                y=points[~inside_points, 1],
                z=points[~inside_points, 2],
                mode="markers",
                marker=dict(size=2, color="red"),
                name="Punkty zewnętrzne",
            )
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data",
            ),
            title="Punkty Monte Carlo",
            margin=dict(l=0, r=0, b=0, t=30),
            height=500,
        )

        if self.plot_container is not None:
            self.plot_container.empty()

        self.plot_container = st.empty()
        self.plot_container.plotly_chart(
            fig, use_container_width=True, key="monte_carlo_plot"
        )

    def plot_cuboids(self, cube_centers, heights):
        fig = go.Figure()

        vertices = self.solid.vertices
        faces = self.solid.faces
        fig.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=0.25,
                color="lightblue",
                name="Bryła",
            )
        )

        def create_cuboid_edges(center, size, height):
            cx, cy, cz = center
            sx, sy = size
            dz = height / 2.0

            dx = sx / 2.0
            dy = sy / 2.0

            zmin = cz - dz
            zmax = cz + dz

            corners = np.array(
                [
                    [cx - dx, cy - dy, zmin], 
                    [cx + dx, cy - dy, zmin], 
                    [cx + dx, cy + dy, zmin], 
                    [cx - dx, cy + dy, zmin], 
                    [cx - dx, cy - dy, zmax], 
                    [cx + dx, cy - dy, zmax],  
                    [cx + dx, cy + dy, zmax],  
                    [cx - dx, cy + dy, zmax],  
                ]
            )

            edges = [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],  # dolna podstawa
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],  # górna podstawa
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],  # krawędzie pionowe
            ]

            x_lines, y_lines, z_lines = [], [], []
            for e in edges:
                x_lines.append(corners[e[0], 0])
                y_lines.append(corners[e[0], 1])
                z_lines.append(corners[e[0], 2])

                x_lines.append(corners[e[1], 0])
                y_lines.append(corners[e[1], 1])
                z_lines.append(corners[e[1], 2])

                x_lines.append(None)
                y_lines.append(None)
                z_lines.append(None)

            return x_lines, y_lines, z_lines

        all_x, all_y, all_z = [], [], []
        for idx, center in enumerate(cube_centers):
            h = heights[idx]
            sx, sy = self.cube_size
            x_l, y_l, z_l = create_cuboid_edges(center, (sx, sy), h)
            all_x.extend(x_l)
            all_y.extend(y_l)
            all_z.extend(z_l)

        fig.add_trace(
            go.Scatter3d(
                x=all_x,
                y=all_y,
                z=all_z,
                mode="lines",
                line=dict(color="green", width=2),
                name="Krawędzie prostopadłościanów",
                hoverinfo="none",
            )
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data",
            ),
            title="Prostopadłościany dopasowane do bryły",
            margin=dict(l=0, r=0, b=0, t=30),
            height=500,
        )

        if self.plot_container is not None:
            self.plot_container.empty()
        self.plot_container = st.empty()
        self.plot_container.plotly_chart(
            fig, use_container_width=True, key="cuboid_plot"
        )


def run_app():
    st.set_page_config(page_title="VolumeCalc", layout="wide")
    st.title("Kalkulator objętości bryły")

    app = App()

    app.load_stl()
    app.display_stl()

    method = st.sidebar.selectbox(
        "Wybierz metodę obliczania objętości",
        options=["Monte Carlo", "Prostokąty"],
        index=0,
    )

    if method == "Monte Carlo":
        app.monte_carlo_ui()
    elif method == "Prostokąty":
        app.cube_ui()

    app.display_info()


if __name__ == "__main__":
    run_app()
