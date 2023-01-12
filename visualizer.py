"""
    FeatureCloud Flimma Application
    Copyright 2023 Mohammad Bakhtiari. All Rights Reserved.
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import os
from dash import dcc
import dash_bio
import numpy as np
import math
from time import sleep, time
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
    global fc_visualization, fig, fig2
    path_prefix_visualizer = os.getenv("PATH_PREFIX") + 'visualizer/'
    print("PATH_PREFIX environment variable: ", path_prefix_visualizer)
    # Start visualization service. It will be available in app frontend url + /visualizer
    fc_visualization = fcvisualization.fcvisualization()
    volcano_plot_style = {'display': 'none'} if len(df) == 0 else {}
    tab_children = [
        dcc.Tab(label='Volcano plot', value='tab-volcano-plot', style=volcano_plot_style),
    ]
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
    print_help_msg()
    # We start the visualization in a thread
    thread_vis = Thread(target=fc_visualization.start, args=(
        'fc', path_prefix_visualizer, callback_fn_terminal_state, extra_visualization_content))
    thread_vis.start()

    # # We start the visualization in a thread
    # thread_vis = Thread(target=fc_visualization.start,
    #                     args=('fc', path_prefix_visualizer, callback_fn_terminal_state, extra_visualization_content))
    # thread_vis.start()
    wait_for_termination()


def wait_for_termination(last_print=time()):
    while True:
        global vis_objects, fig, fig2
        # When the callback function will fire in the visualizer app, it'll trigger Finished state
        if TERMINAL is True:
            print('plot is finished')
            break

        if time() > last_print + 10:
            print_help_msg()
            last_print = time()
        sleep(2)


def print_help_msg():
    print("The volcano plot is available in the app's front-end. You can have access to the plot by taking the "
          "following steps:\n "
          "\t1. Click on the front-end button.\n"
          "\t2. Click on the `View in the new tab` button.\n"
          "\t3. In the new tab, add `/visualizer` to the URL and press Enter.\n"
          "\t4. Go to the `Volcano` tab to view the plot.\n"
          "To finish the app execution, please click on the `Finish` button at the right corner.")
