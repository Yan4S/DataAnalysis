from collections import namedtuple

import numpy as np
import pandas as pd
import joblib

Model = namedtuple('Model', ['model', 'model_name', 'cv_score', 'test_score'])
class ModelSaver:
    
    def __init__(self, filename='models.joblib'):
        self.models = []
        self.filename = filename
