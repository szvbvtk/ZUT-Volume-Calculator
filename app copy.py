import streamlit as st
import numpy as np
from stl import mesh
import plotly.graph_objects as go
from io import BytesIO


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

        if stl_mesh is not None:
            self.process_stl()

    def process_stl(self):
        self.vertices = self.stl_mesh.vectors.reshape(-1, 3)
        self.faces = np.arange(len(self.vertices)).reshape(-1, 3)
        # self.faces = self.mesh.vectors

    def is_point_inside(self, point):
        direction_vector = np.array([1, 0, 0])
        intersection_count = 0

        for i in range(len(self.faces)):
            v0 = self.vertices[self.faces[i][0]]
            v1 = self.vertices[self.faces[i][1]]
            v2 = self.vertices[self.faces[i][2]]

            if ray_intersects_triangle(point, direction_vector, v0, v1, v2):
                intersection_count += 1

        is_inside = intersection_count % 2 == 1
        return is_inside

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


class App:
    def __init__(self):
        self.solid = None

    def load_stl(self):
        st.sidebar.title("Wczytaj plik STL")
        uploaded_file = st.sidebar.file_uploader("Wybierz plik STL", type=["stl"])

        if uploaded_file is not None:
            stl_mesh = mesh.Mesh.from_file(
                uploaded_file.name, fh=BytesIO(uploaded_file.read())
            )
            self.solid = Solid(stl_mesh)
            st.success("Plik STL został wczytany pomyślnie.")
        else:
            st.info("Proszę wczytać plik STL.")

    def display_stl(self):
        if self.solid is None:
            st.warning("Nie wczytano pliku STL.")
            return

        fig = go.Figure()

        vertices = self.solid.vertices
        faces = self.solid.faces

        # Create a mesh3d plot
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
                # xaxis=dict(title='X'),
                # yaxis=dict(title='Y'),
                # zaxis=dict(title='Z'),
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data",
            ),
            title="Wizualizacja STL",
        )

        st.plotly_chart(fig, use_container_width=True)

    def display_info(self):
        if self.solid is None:
            return

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

        num_points = st.sidebar.slider(
            "Liczba punktów Monte Carlo",
            min_value=10,
            max_value=100000,
            value=10000,
            step=100,
        )

        if st.sidebar.button("Oblicz objętość"):
            estimator = MonteCarloVolumeEstimator(self.solid, num_points)
            volume = estimator.run()

            st.sidebar.success(f"Przybliżona objętość: {volume:.2f} jednostek^3")

            points, inside_points = estimator.get_points()
            self.plot_monte_carlo(points, inside_points)

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

        st.plotly_chart(fig, use_container_width=True)


def run_app():
    st.set_page_config(page_title="Wizualizacja STL", layout="wide")
    st.title("Wizualizacja STL")

    app = App()
    app.load_stl()
    app.display_stl()
    app.display_info()
    app.monte_carlo_ui()


if __name__ == "__main__":
    run_app()
