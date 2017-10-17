from task2.optimization import conjugate_gradients
from task2.presubmit_tests import TestCG, TestHFN

if __name__ == "__main__":
    test_cg = TestCG()
    test_cg.test_default()
    test_cg.test_display()
    test_cg.test_histories()
    test_cg.test_max_iter()
    test_cg.test_tolerance()

    test_hfn = TestHFN()
    test_hfn.test_default()
    test_hfn.test_tolerance()
    test_hfn.test_max_iter()
    test_hfn.test_display()
    test_hfn.test_history()
    test_hfn.test_line_search_options()
    test_hfn.test_quality()