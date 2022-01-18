import numpy as np
import pandas as pd
from FeatureCloud.engine.app import LogLevel, app
from FeatureCloud.api.http_ctrl import api_server
from FeatureCloud.api.http_web import web_server
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
