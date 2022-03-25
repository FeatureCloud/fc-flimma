""""
    FeatureCloud Flimma Application
    Copyright 2022 Mohammad Bakhtiari. All Rights Reserved.
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
import numpy as np
import pandas as pd
from FeatureCloud.app.engine.app import LogLevel, app
from FeatureCloud.app.api.http_ctrl import api_server
from FeatureCloud.app.api.http_web import web_server
from bottle import Bottle


def run(host='localhost', port=5000):
    """ run the docker container on specific host and port.

    Parameters
    ----------
    host: str
    port: int

    """

    app.register()
    server = Bottle()
    server.mount('/api', api_server)
    server.mount('/web', web_server)
    server.run(host=host, port=port)


def readfiles(flimma_counts_file_path, flimma_design_file_path):
    counts_df = pd.read_csv(flimma_counts_file_path, index_col=0, sep="\t")
    design_df = pd.read_csv(flimma_design_file_path, index_col=0, sep="\t")
    return counts_df, design_df


class JsonSerializer:
    """
    A serilizer to automatically convert all NumPy arrays, Panda DataFrames, and Pandas Series
    in a nested data structure into lists. All list, tuples, and dictionaries in the submitted data
    will remain untouched
    """

    def __init__(self):
        self.encoder = {pd.DataFrame: lambda data: data.to_numpy().tolist(),
                        pd.core.series.Series: lambda data: data.tolist(),
                        np.ndarray: lambda data: self.encode_numpy(data),
                        dict: lambda data: self.encode_dict(data),
                        list: lambda data: self.encode_list(data)}
        # self.encoder = {pd.DataFrame: pd.DataFrame.to_numpy,
        #                 pd.core.series.Series: pd.Series.tolist,
        #                 np.ndarray: self.encode_numpy,
        #                 dict: self.encode_dict,
        #                 list: self.encode_list}
    def prepare(self, data):
        if type(data) in self.encoder.keys():
            return self.encoder[type(data)](data)
        return data

    def encode_list(self, data):
        l = []
        for item in data:
            l.append(self.prepare(item))
        return l

    def encode_dict(self, data):
        return {k: self.prepare(v) for k, v in data.items()}

    def encode_numpy(self, data):
        l = []
        for item in data.tolist():
            l.append(self.prepare(item))
        return l

js_serializer = JsonSerializer()