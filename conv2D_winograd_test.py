from log_utils import get_logger
from log_utils import set_up_logging
import logging
import unittest
import numpy as np
import torch
from torch import tensor
from conv2D_winograd import Winograd

"""
author: Adam Dziedzic ady@uchicago.edu
"""


class TestPyTorchConv1d(unittest.TestCase):

    def setUp(self):
        log_file = "pytorch_conv2D_winograd.log"
        is_debug = True
        set_up_logging(log_file=log_file, is_debug=is_debug)
        self.logger = get_logger(name=__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("Set up test")

    def testSimpleWinograd(self):
        x = tensor([[[[1.0, 2.0, 3.0, -1.0],
                      [3.0, 4.0, 1.0, 2.0],
                      [1.0, 2.0, 1.0, -2.0],
                      [2.0, 1.0, -1.0, 2.0]]]])
        # A single filter.
        y = tensor([[[[1.0, 2.0, -1.0],
                      [3.0, 2.0, 1.0],
                      [4.0, 1.0, -2.0]]]])
        expect = torch.nn.functional.conv2d(x, y)
        result = Winograd.winograd_F_2_3(x, y)
        np.testing.assert_array_almost_equal(
            x=expect, y=result,
            err_msg="The expected array x and computed y are not almost equal.")

    def testWinograd(self):
        x = torch.randint(-3, 3, (2, 3, 6, 6), dtype=torch.float)
        y = torch.randint(-3, 3, (3, 3, 3, 3), dtype=torch.float)
        expect = torch.nn.functional.conv2d(x, y)
        result = Winograd.forward(x, y)
        np.testing.assert_array_almost_equal(
            x=expect, y=result,
            err_msg="The expected array x and computed y are not almost equal.")


if __name__ == '__main__':
    unittest.main()
