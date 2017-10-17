from task1.optimization import gradient_descent, newton
from task1.oracles import QuadraticOracle, create_log_reg_oracle
from sklearn.datasets import load_svmlight_file
from task1.presubmit_tests import *

import numpy as np

if __name__ == "__main__":
    A_gisette, b_gisette = load_svmlight_file("gisette_scale")
    logreg_gisette = create_log_reg_oracle(A_gisette, b_gisette, regcoef=1 / A_gisette.shape[0])
    x0 = np.ones(A_gisette.shape[0])
    logreg_gisette.matmat_ATsA(x0)