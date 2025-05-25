import streamlit as st
import numpy as np
from stl import mesh
import plotly.graph_objects as go
from io import BytesIO
import time
from uuid import uuid4


def ray_intersects_triangle(orig, dir, v0, v1, v2):
    EPSILON = 1e-8
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(dir, edge2)
    a = np.dot(edge1, h)

    if -EPSILON < a < EPSILON:
        return False

    f = 1.0 / a
    s = orig - v0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return False

    q = np.cross(s, edge1)
    v = f * np.dot(dir, q)
    if v < 0.0 or u + v > 1.0:
        return False

    t = f * np.dot(edge2, q)
    return t > EPSILON


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
        # self.faces = self.mesh.vectors

        tris = self.vertices[self.faces]
        self.triangle_bboxes = np.empty((len(self.faces), 2, 3))
        self.triangle_bboxes[:, 0, :] = np.min(tris, axis=1)
        self.triangle_bboxes[:, 1, :] = np.max(tris, axis=1)

    def is_point_inside(self, point):
        direction_vector = np.array([1, 0, 0])
        intersection_count = 0

        y_in_range = (point[1] >= self.triangle_bboxes[:, 0, 1]) & (
            point[1] <= self.triangle_bboxes[:, 1, 1]
        )
        z_in_range = (point[2] >= self.triangle_bboxes[:, 0, 2]) & (
            point[2] <= self.triangle_bboxes[:, 1, 2]
        )
        candidates = np.where(y_in_range & z_in_range)[0]

        for i in candidates:
            v0 = self.vertices[self.faces[i][0]]
            v1 = self.vertices[self.faces[i][1]]
            v2 = self.vertices[self.faces[i][2]]

            if ray_intersects_triangle(point, direction_vector, v0, v1, v2):
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
                    "X": [np.min(self.vertices[:, 0]), np.max(self.vertices[:, 0])],
                    "Y": [np.min(self.vertices[:, 1]), np.max(self.vertices[:, 1])],
                    "Z": [np.min(self.vertices[:, 2]), np.max(self.vertices[:, 2])],
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

    def run(self):
        self.generate_random_points()

        self.inside_points = self.solid.contains(self.points)

        bounding_box_volume = np.prod(self.bounding_box[1] - self.bounding_box[0])

        volume_estimate = (
            np.sum(self.inside_points) / self.num_points
        ) * bounding_box_volume

        return volume_estimate

    def get_points(self):
        return self.points, self.inside_points

    def run_with_progress(self, progress_callback=None):
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


class CubeVolumeEstimator:
    def __init__(self, solid, cube_size=1):
        self.solid = solid
        self.cube_size = cube_size
        self.bounding_box = self.solid.vertices.min(axis=0), self.solid.vertices.max(
            axis=0
        )
        self.inside_cubes = None
        self.cube_centers = None

    def run(self, progress_callback=None):
        min_bounds, max_bounds = self.bounding_box
        cube_size = self.cube_size
        cube_volume = cube_size**3

        x_vals = np.arange(min_bounds[0], max_bounds[0], cube_size)
        y_vals = np.arange(min_bounds[1], max_bounds[1], cube_size)
        z_vals = np.arange(min_bounds[2], max_bounds[2], cube_size)

        total_cubes = len(x_vals) * len(y_vals) * len(z_vals)
        inside_count = 0
        checked_cubes = 0
        inside = np.zeros(total_cubes, dtype=bool)
        centers = []  # Lista do przechowywania centrów sześcianów

        for xi, x in enumerate(x_vals):
            for yi, y in enumerate(y_vals):
                for zi, z in enumerate(z_vals):
                    cube_center = np.array(
                        [x + cube_size / 2, y + cube_size / 2, z + cube_size / 2]
                    )
                    inside[checked_cubes] = self.solid.is_point_inside(cube_center)
                    centers.append(cube_center)

                    if inside[checked_cubes]:
                        inside_count += 1

                    checked_cubes += 1

                    if (
                        progress_callback
                        and checked_cubes % max(1, total_cubes // 100) == 0
                    ):
                        progress_callback(checked_cubes / total_cubes)

        volume_estimate = inside_count * cube_volume
        self.inside_cubes = inside
        self.cube_centers = np.array(centers)  # Zapis centrów sześcianów
        return volume_estimate

    def get_cubes(self):
        return self.cube_centers, self.inside_cubes


class App:
    def __init__(self):
        self.solid = None
        self.plot_container = None
        self.default_plot = None
        self.cube_size = 1.0

    def set_computed_flag(self, val):
        st.session_state.is_monte_carlo_computed = val

    def load_stl(self):
        st.sidebar.title("Wczytaj plik STL")
        uploaded_file = st.sidebar.file_uploader("Wybierz plik STL", type=["stl"])

        if uploaded_file is not None:
            stl_mesh = mesh.Mesh.from_file(
                uploaded_file.name, fh=BytesIO(uploaded_file.read())
            )
            self.solid = Solid(stl_mesh)
            # st.success("Plik STL został wczytany pomyślnie.")
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
            st.warning("Nie wczytano pliku STL.")
            return

        st.sidebar.subheader("metoda Monte Carlo")

        num_points = st.sidebar.number_input(
            "Liczba punktów Monte Carlo",
            min_value=10,
            max_value=100000,
            value=1000,
            step=100,
            format="%d",
            help="Liczba punktów Monte Carlo do oszacowania objętości.",
        )

        col1, col2 = st.sidebar.columns(2, gap="small")

        with col1:
            calculate_button = st.button(
                "Oblicz objętość",
                key="calculate_button",
                on_click=self.set_computed_flag,
                args=(True,),
            )

        reset_button = False
        if st.session_state.is_monte_carlo_computed:
            with col2:
                reset_button = st.button(
                    "Resetuj wyniki",
                    key="reset_button",
                    on_click=self.set_computed_flag,
                    args=(False,),
                )

        if reset_button:
            if self.plot_container is not None:
                self.plot_container.empty()

            self.plot_container.plotly_chart(
                self.default_plot, use_container_width=True, key=f"plot_{uuid4()}"
            )

            st.sidebar.info("Wyniki zostały zresetowane.")

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

            volume = estimator.run_with_progress(progress_callback=update_progress)
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

        st.sidebar.subheader("metoda sześcianów")

        self.cube_size = st.sidebar.number_input(
            "Rozmiar sześcianu",
            min_value=0.01,
            max_value=50.0,
            value=1.0,
            step=0.01,
            format="%.2f",
            help="Rozmiar sześcianu do oszacowania objętości.",
        )

        calculate_button = st.sidebar.button("Oblicz objętość", key="calculate_button2")

        if calculate_button:
            if self.plot_container is not None:
                self.plot_container.empty()

            start_time = time.time()
            estimator = CubeVolumeEstimator(self.solid, self.cube_size)

            def update_progress(pct):
                progress_bar.progress(
                    min(int(pct * 100), 100), text=f"Obliczanie... {int(pct * 100)}%"
                )

            progress_bar = st.progress(0, text="Obliczanie objętości...")

            volume = estimator.run(progress_callback=update_progress)
            cubes, inside_cubes = estimator.get_cubes()
            self.plot_cubes(cubes, inside_cubes)

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
        )

        if self.plot_container is not None:
            self.plot_container.empty()

        self.plot_container = st.empty()
        self.plot_container.plotly_chart(
            fig, use_container_width=True, key="monte_carlo_plot"
        )

    def plot_cubes(self, cube_centers, inside_cubes):
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

        def create_cube_lines(x, y, z, size):
            r = [-0.5, 0.5]
            vertices = [
                [x + r[i] * size, y + r[j] * size, z + r[k] * size]
                for i in range(2)
                for j in range(2)
                for k in range(2)
            ]
            edges = [
                [0, 1],
                [1, 3],
                [3, 2],
                [2, 0],  # dół
                [4, 5],
                [5, 7],
                [7, 6],
                [6, 4],  # góra
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],  # boki
            ]
            x_lines, y_lines, z_lines = [], [], []
            for edge in edges:
                for idx in [0, 1]:
                    x_lines.append(vertices[edge[idx]][0])
                    y_lines.append(vertices[edge[idx]][1])
                    z_lines.append(vertices[edge[idx]][2])
                x_lines.append(None)  # Przerwa między krawędziami
                y_lines.append(None)
                z_lines.append(None)
            return x_lines, y_lines, z_lines

        inside_x, inside_y, inside_z = [], [], []
        outside_x, outside_y, outside_z = [], [], []
        for i, center in enumerate(cube_centers):
            x_lines, y_lines, z_lines = create_cube_lines(
                center[0], center[1], center[2], self.cube_size
            )
            if inside_cubes[i]:
                inside_x.extend(x_lines)
                inside_y.extend(y_lines)
                inside_z.extend(z_lines)
            else:
                outside_x.extend(x_lines)
                outside_y.extend(y_lines)
                outside_z.extend(z_lines)

        if inside_x:
            fig.add_trace(
                go.Scatter3d(
                    x=inside_x,
                    y=inside_y,
                    z=inside_z,
                    mode="lines",
                    line=dict(color="green", width=2),
                    name="Wewnętrzne sześciany",
                    hoverinfo="none",
                )
            )

        if outside_x:
            fig.add_trace(
                go.Scatter3d(
                    x=outside_x,
                    y=outside_y,
                    z=outside_z,
                    mode="lines",
                    line=dict(color="red", width=1),
                    opacity=0.3,
                    name="Zewnętrzne sześciany",
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
            title="Sześciany wewnątrz i na zewnątrz bryły",
        )

        if self.plot_container is not None:
            self.plot_container.empty()

        self.plot_container = st.empty()
        self.plot_container.plotly_chart(
            fig, use_container_width=True, key="cubes_plot"
        )


def run_app():
    st.set_page_config(page_title="VolumeCalc", layout="wide")
    st.title("Kalkulator objętości bryły")

    app = App()

    if "is_monte_carlo_computed" not in st.session_state:
        st.session_state.is_monte_carlo_computed = False

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
