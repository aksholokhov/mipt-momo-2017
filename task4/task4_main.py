import numpy as np
from scipy import sparse

from task4.oracles import create_lasso_nonsmooth_oracle, create_lasso_prox_oracle, L1RegOracle
from task4.presubmit_tests import test_l1_reg_oracle, test_l1_reg_oracle_2, test_lasso_nonsmooth_oracle, test_lasso_prox_oracle
from task4.presubmit_tests import test_lasso_duality_gap, test_least_squares_oracle, test_least_squares_oracle_2
from task4.presubmit_tests import test_subgradient_prototype, test_subgradient_one_step, test_subgradient_one_step_nonsmooth
from task4.presubmit_tests import test_proximal_gd_prototype, test_proximal_gd_one_step, test_proximal_nonsmooth, test_proximal_nonsmooth2
from task4.presubmit_tests import test_accelerated_proximal_gd_prototype, test_accelerated_proximal_nonsmooth, test_accelerated_proximal_nonsmooth2
from task4.optimization import subgradient_method, proximal_gradient_descent, accelerated_proximal_gradient_descent
from sklearn.datasets import load_svmlight_file

if __name__ == "__main__":
    # test_l1_reg_oracle()
    # test_l1_reg_oracle_2()
    # test_lasso_nonsmooth_oracle()
    # test_lasso_prox_oracle()
    # test_lasso_duality_gap()
    # test_least_squares_oracle()
    # test_least_squares_oracle_2()
    # test_subgradient_prototype()
    # test_subgradient_one_step()
    # test_subgradient_one_step_nonsmooth()
    # test_proximal_gd_prototype()
    # test_proximal_gd_one_step()
    # test_proximal_nonsmooth()
    # test_proximal_nonsmooth2()
    # test_accelerated_proximal_gd_prototype()
    # test_accelerated_proximal_nonsmooth()
    # test_proximal_nonsmooth2()


    A_rs, b_rs = load_svmlight_file("../datasets/real-sim")
    b_rs = sparse.csr_matrix(b_rs)
    rs_prox = create_lasso_prox_oracle(A_rs, b_rs, regcoef=1 / A_rs.shape[0])
    rs_sub = create_lasso_nonsmooth_oracle(A_rs, b_rs, regcoef=1 / A_rs.shape[0])
    x_0 = sparse.rand(1, A_rs.shape[1], density=0.1)

    [x_star, status, hist_sub] = subgradient_method(rs_sub, x_0, trace=True, max_iter=10, tolerance=1e-6)