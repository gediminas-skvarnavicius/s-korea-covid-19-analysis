# pylint: disable=import-error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Sequence, Union
import numpy as np
import numpy.typing as npt
from datetime import datetime


def two_yaxis_plotly(
    x_values: Sequence[float],
    y1_values: Sequence[float],
    y2_values: Sequence[float],
    y1_title: str,
    y2_title: str,
    x_title: str,
    colors: Sequence[str],
    size: Sequence[str],
    yrange: Optional[Sequence[float]],
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

    if yrange:
        fig.update_layout(yaxis=dict(range=yrange))

    return fig


def add_alert_background(fig: go.Figure, orange: bool = True, red: bool = True) -> None:
    """Adds shapes to a plotly figure that correspond to infectious disease alert levels"""
    if red:
        fig.add_shape(
            type="rect",
            x0="2020-02-23",  # start x-value of rectangle
            x1="2020-07-05",  # end x-value of rectangle
            y0=fig.layout.yaxis.range[0],  # start y-value of rectangle
            y1=fig.layout.yaxis.range[1],  # end y-value of rectangle
            fillcolor="rgba(255, 0, 0, 0.2)",  # fill color with opacity
            line=dict(width=0),  # set the border width to 0 to remove the border
        )

    if orange:
        fig.add_shape(
            type="rect",
            x0="2020-01-28",  # start x-value of rectangle
            x1="2020-02-23",  # end x-value of rectangle
            y0=fig.layout.yaxis.range[0],  # start y-value of rectangle
            y1=fig.layout.yaxis.range[1],  # end y-value of rectangle
            fillcolor="rgba(255, 165, 0, 0.2)",  # fill color with opacity
            line=dict(width=0),  # set the border width to 0 to remove the border
        )


def closest_point_plotly(
    fig: go.Figure,
    value: Union[float, datetime],
    val_to_get: str = "y",
    data_no: int = 0,
) -> float:
    """Gets the closest x or y value to x or y argument"""
    assert val_to_get in ["x", "y"], "val to get must be x or y"
    if val_to_get == "y":
        x_vals: npt.NDArray = fig.data[data_no].x
        closest_xi: Union[float, datetime] = np.abs(x_vals - value).argmin()
        closest_point = fig.data[0].y[closest_xi]
    if val_to_get == "x":
        y_vals: npt.NDArray = fig.data[data_no].y
        closest_yi: Union[float, datetime] = np.abs(y_vals - value).argmin()
        closest_point = fig.data[0].y[closest_yi]
    return closest_point


def annotate_plotly_by_val(
    fig: go.Figure,
    value: Union[float, datetime],
    text: str,
    val_ax: str = "x",
    data_no: int = 0,
    **kwargs,
) -> None:
    """Annotates the given figure by single x or y value"""
    assert val_ax in ["x", "y"], "val to get must be x or y"
    if val_ax == "x":
        y_val = closest_point_plotly(fig, value, val_to_get="y", data_no=data_no)
        fig.add_annotation(
            x=value,  # x-coordinate of text position
            y=y_val,  # y-coordinate of text position
            xref="x",  # x-coordinate reference
            yref="y",  # y-coordinate reference
            text=text,  # text to display
            **kwargs,
        )
    if val_ax == "y":
        x_val = closest_point_plotly(fig, value, val_to_get="x", data_no=data_no)
        fig.add_annotation(
            x=x_val,  # x-coordinate of text position
            y=value,  # y-coordinate of text position
            xref="x",  # x-coordinate reference
            yref="y",  # y-coordinate reference
            text=text,  # text to display
            **kwargs,
        )
