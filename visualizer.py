import os
from dash import Dash, dcc, html
import dash_bio
import numpy as np
import math
from time import sleep
from threading import Thread

import fcvisualization

TERMINAL = False
fc_visualization = None
fig = None
fig2 = None
vis_objects = []


def callback_fn_terminal_state():
    global TERMINAL
    print("Transition to terminal state triggered...")
    TERMINAL = True


def plot_volcano(df):
    print(df.shape)
    print(df.columns)
    print(df.index.values[:5])
    print(df.head(1))
    global fc_visualization, fig, fig2
    path_prefix_visualizer = os.getenv("PATH_PREFIX") + 'visualizer/'
    print("PATH_PREFIX environment variable: ", path_prefix_visualizer)
    # Start visualization service. It will be available in app frontend url + /visualizer
    fc_visualization = fcvisualization.fcvisualization()
    volcano_plot_style = {'display': 'none'} if len(df) == 0 else {}
    tab_children = [
        dcc.Tab(label='Volcano plot', value='tab-volcano-plot', style=volcano_plot_style),
    ]
    tab_value = 'tab-volcano-plot'
    show_toast = False
    min_effect = math.floor(df['EFFECTSIZE'].min())
    max_effect = math.ceil(df['EFFECTSIZE'].max())
    min_effect_value = math.floor(min_effect + 0.3 * (max_effect - min_effect))
    max_effect_value = math.ceil(max_effect - 0.3 * (max_effect - min_effect))

    min_p_value = -math.floor(np.log10(df['P'].min()))
    max_p_value = -math.ceil(np.log10(df['P'].max()))
    min_genome_wide_line = min(min_p_value, max_p_value)
    max_genome_wide_line = max(min_p_value, max_p_value)
    genome_wide_line_value = math.floor(min_genome_wide_line + 0.3 * (max_genome_wide_line - min_genome_wide_line))

    fig = dash_bio.VolcanoPlot(
        dataframe=df,
        genomewideline_value=genome_wide_line_value,
    )
    extra_visualization_content = [{
        "title": "Volcano",
        "fig": fig,
    }]
    print('Plot start...')
    # We start the visualization in a thread
    thread_vis = Thread(target=fc_visualization.start, args=(
        'fc', path_prefix_visualizer, callback_fn_terminal_state, extra_visualization_content))
    thread_vis.start()

    print('Plot start...')
    # We start the visualization in a thread
    thread_vis = Thread(target=fc_visualization.start,
                        args=('fc', path_prefix_visualizer, callback_fn_terminal_state, extra_visualization_content))
    thread_vis.start()
    wait_for_termination()


def wait_for_termination():
    while (True):
        global vis_objects, fig, fig2
        # When the callback function will fire in the visualizer app, it'll trigger Finished state
        if TERMINAL is True:
            print('plot is finished')
            break

        original_title = "My Diagram from State machine"
        if len(vis_objects) == 2:
            # Update the diagram added in the previous iteration
            for diagram in vis_objects:
                if diagram['title'] == original_title:
                    print('Update diagram in progress')
                    diagram['title'] = original_title + ' updated'
                    diagram['fig'] = fig2
                    vis_objects = fc_visualization.update_diagram(diagram)
            # Add a one more diagram
            vis_objects = fc_visualization.add_diagram([{
                "title": "My second diagram from state machine",
                "fig": fig2,
            }])

        if len(vis_objects) == 0:
            # Add a new diagram to the UI
            print("Adding a new diagram to the UI")
            vis_objects = fc_visualization.add_diagram([{
                "title": original_title,
                "fig": fig,
            }])

        print(f'Visualization objects ==> {vis_objects}')
        print('plot is running')
        sleep(2)