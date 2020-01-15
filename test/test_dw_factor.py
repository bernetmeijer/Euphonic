import unittest
import numpy as np
import numpy.testing as npt
from euphonic.data.phonon import PhononData
from euphonic.data.interpolation import InterpolationData

class TestDWFactorLZO(unittest.TestCase):

    def setUp(self):
        seedname = 'La2Zr2O7-grid'
        path = 'test/data/structure_factor/LZO'
        self.data = PhononData.from_castep(seedname, path=path)
        self.dw_path = 'test/data/dw_factor/LZO/' 

    def test_dw_T5(self):
        dw = self.data._dw_coeff(5)
        expected_dw = np.reshape(np.loadtxt(self.dw_path + 'dw_T5.txt'),
                                 (self.data.n_ions, 3, 3))

        npt.assert_allclose(dw, expected_dw)

    def test_dw_T100(self):
        dw = self.data._dw_coeff(100)
        expected_dw = np.reshape(np.loadtxt(self.dw_path + 'dw_T100.txt'),
                                 (self.data.n_ions, 3, 3))

        npt.assert_allclose(dw, expected_dw)

class TestDWFactorQuartz(unittest.TestCase):

    def setUp(self):
        seedname = 'quartz'
        path = 'test/data/interpolation/quartz'
        self.data = InterpolationData.from_castep(seedname, path=path)
        qpts = np.loadtxt('test/data/qgrid_444.txt')
        self.data.calculate_fine_phonons(qpts, asr='reciprocal')
        self.dw_path = 'test/data/dw_factor/quartz/' 

    def test_dw_T5(self):
        dw = self.data._dw_coeff(5)
        expected_dw = np.reshape(np.loadtxt(self.dw_path + 'dw_T5.txt'),
                                 (self.data.n_ions, 3, 3))

        npt.assert_allclose(dw, expected_dw)

    def test_dw_T100(self):
        dw = self.data._dw_coeff(100)
        expected_dw = np.reshape(np.loadtxt(self.dw_path + 'dw_T100.txt'),
                                 (self.data.n_ions, 3, 3))

        npt.assert_allclose(dw, expected_dw)
