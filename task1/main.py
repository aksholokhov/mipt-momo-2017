from task1.optimization import gradient_descent, newton
from task1.oracles import QuadraticOracle
from task1.presubmit_tests import *

import numpy as np

if __name__ == "__main__":
    # oracle = QuadraticOracle(np.eye(5), np.arange(5) + 8*np.ones(5))
    # x_opt, message, history = newton(oracle, np.zeros(5), line_search_options={'method': 'Constant', 'c': 1.0})
    # print('Found optimal point: {}'.format(x_opt))
    test_log_reg_usual()
    test_log_reg_oracle_calls()
