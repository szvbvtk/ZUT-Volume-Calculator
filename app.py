import streamlit as st
import numpy as np
from stl import mesh
import plotly.graph_objects as go
from abc import ABC, abstractmethod
from io import BytesIO


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


def run_app():
    st.set_page_config(page_title="Wizualizacja STL", layout="wide")
    st.title("Wizualizacja STL")

    app = App()
    app.load_stl()
    app.display_stl()
    app.display_info()


if __name__ == "__main__":
    run_app()
