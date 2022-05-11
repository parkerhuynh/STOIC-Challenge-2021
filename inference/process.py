from typing import Dict
from pathlib import Path
import SimpleITK
from tensorflow.keras import models

import numpy as np
import os
from os import listdir
from os.path import isfile, join

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

from utils import MultiClassAlgorithm, to_input_format, unpack_single_output, device
from algorithm.preprocess import preprocess


COVID_OUTPUT_NAME = Path("probability-covid-19")
SEVERE_OUTPUT_NAME = Path("probability-severe-covid-19")


class StoicAlgorithm(MultiClassAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            input_path=Path("/input/images/ct/"),
            output_path=Path("/output/")
        )
    def predict(self, *, input_image: SimpleITK.Image) -> Dict:
        # pre-processing
        input_image = preprocess(input_image)
        input_image = np.array([input_image])

        #Get models
        mypath = "./artifact/"
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        severe_files = [f for f in onlyfiles if f[-9:] == "Severe.h5"]
        covid_files =  [f for f in onlyfiles if f[-8:] == "COVID.h5"]
        
        #load model and predictcd 
        prob_covid = 0
        for covid_file in covid_files:
            covid_model = models.load_model(f"./artifact/{covid_file}")
            prob_covid_i = covid_model.predict(input_image)[0][0].astype(float)
            prob_covid += prob_covid_i
        prob_covid =  prob_covid/len(covid_files)
        
        prob_severe = 0
        for severe_file in severe_files:
            severe_model = models.load_model(f"./artifact/{severe_file}")
            prob_severe_i = severe_model.predict(input_image)[0][0].astype(float)
            prob_severe += prob_severe_i
        prob_severe =  prob_severe/len( severe_files)
        
        return {
            COVID_OUTPUT_NAME: prob_covid,
            SEVERE_OUTPUT_NAME: prob_severe
        }


if __name__ == "__main__":
    StoicAlgorithm().process()
