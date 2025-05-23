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

    # def contains(self, points):
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         results = list(executor.map(self.is_point_inside, points))

        return np.array(results, dtype=bool)

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
            # aktualizuj pasek co pewien procent
            if progress_callback and i % max(1, total // 100) == 0:
                progress_callback(i / total)

        self.inside_points = inside

        bounding_box_volume = np.prod(self.bounding_box[1] - self.bounding_box[0])
        volume_estimate = (np.sum(self.inside_points) / self.num_points) * bounding_box_volume

        return volume_estimate



class App:
    def __init__(self):
        self.solid = None
        self.plot_container = None
        self.default_plot = None

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
            calculate_button = st.button("Oblicz objętość", key="calculate_button", on_click=self.set_computed_flag, args=(True,))

        reset_button = False
        if st.session_state.is_monte_carlo_computed:
            with col2:
                reset_button = st.button("Resetuj wyniki", key="reset_button", on_click=self.set_computed_flag, args=(False,))

        if reset_button:       
            if self.plot_container is not None:
                self.plot_container.empty()

            self.plot_container.plotly_chart(self.default_plot, use_container_width=True, key=f"plot_{uuid4()}")

            st.sidebar.info("Wyniki zostały zresetowane.")

        if calculate_button:
            if self.plot_container is not None:
                self.plot_container.empty()

            def update_progress(pct):
                progress_bar.progress(min(int(pct * 100), 100), text=f"Obliczanie... {int(pct * 100)}%")

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
        self.plot_container.plotly_chart(fig, use_container_width=True, key="monte_carlo_plot")


def run_app():
    st.set_page_config(page_title="VolumeCalc", layout="wide")
    st.title("Kalkulator objętości bryły")

    app = App()

    if "is_monte_carlo_computed" not in st.session_state:
        st.session_state.is_monte_carlo_computed = False

    app.load_stl()
    app.display_stl()
    app.monte_carlo_ui()
    app.display_info()




if __name__ == "__main__":
    run_app()
