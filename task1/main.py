from task1.optimization import gradient_descent, newton
from task1.oracles import QuadraticOracle
from task1.presubmit_tests import *

import numpy as np

if __name__ == "__main__":
    test_hess_finite_diff_1()
    test_hess_finite_diff_2()