# pylint: disable=import-error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Iterable
import numpy.typing as npt


def two_yaxis_plotly(
    x_values: npt.ArrayLike,
    y1_values: npt.ArrayLike,
    y2_values: npt.ArrayLike,
    y1_title: str,
    y2_title: str,
    x_title: str,
    colors: npt.ArrayLike,
    size: npt.ArrayLike,
) -> go.Figure:
    """Creates a graph with two separate lines sharing the x axis but having different y axes."""
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=x_values, y=y1_values, name=y1_title, line=dict(color=colors[0])),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=x_values, y=y2_values, name=y2_title, line=dict(color=colors[1])),
        secondary_y=True,
    )

    # Set x-axis title
    fig.update_xaxes(title_text=x_title)

    # Set y-axes titles
    fig.update_yaxes(
        title_text=f"<b>{y1_title}</b>", secondary_y=False, patch=dict(color=colors[0])
    )
    fig.update_yaxes(
        title_text=f"<b>{y2_title}</b>", secondary_y=True, patch=dict(color=colors[1])
    )

    fig.update_layout(
        width=size[0],
        height=size[1],
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=10),
    )
    return fig
