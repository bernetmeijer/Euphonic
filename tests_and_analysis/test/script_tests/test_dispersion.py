import sys
import os
import json
from unittest.mock import patch

import pytest
import numpy as np
import numpy.testing as npt
# Required for mocking
import matplotlib.pyplot

from tests_and_analysis.test.utils import get_data_path
from tests_and_analysis.test.script_tests.utils import (
    get_script_test_data_path, get_current_plot_lines_xydata, args_to_key)
import euphonic.cli.dispersion


cahgo2_fc_file = os.path.join(
    get_data_path(), 'force_constants', 'CaHgO2', 'mp-7041-20180417.yaml')
lzo_fc_file = os.path.join(
    get_data_path(), 'force_constants', 'LZO', 'LZO_force_constants.json')
nacl_fc_file = os.path.join(
    get_data_path(), 'force_constants', 'NaCl_cli_test',
    'force_constants.hdf5')
nacl_phonon_file = os.path.join(
    get_data_path(), 'qpoint_phonon_modes', 'NaCl', 'band', 'band.yaml')
nacl_phonon_hdf5_file = os.path.join(
    get_data_path(), 'qpoint_phonon_modes', 'NaCl', 'band', 'band.hdf5')
quartz_phonon_file = os.path.join(
    get_data_path(), 'qpoint_phonon_modes', 'quartz',
    'quartz_bandstructure_qpoint_phonon_modes.json')
disp_output_file = os.path.join(get_script_test_data_path(), 'dispersion.json')
disp_params =  [
    [cahgo2_fc_file],
    [cahgo2_fc_file, '--energy-unit=hartree'],
    [cahgo2_fc_file, '--x-label=wavenumber', '--y-label=Energy (meV)',
     '--title=CaHgO2'],
    [cahgo2_fc_file, '--reorder'],
    [cahgo2_fc_file, '-u=1/cm', '--e-min=200'],
    [cahgo2_fc_file, '--e-min=30', '--e-max=100'],
    [cahgo2_fc_file, '--length-unit=bohr', '--q-distance=0.04'],
    [cahgo2_fc_file, '--q-distance=0.02'],
    [cahgo2_fc_file, '--asr'],
    [cahgo2_fc_file, '--asr=realspace'],
    [lzo_fc_file],
    [nacl_fc_file],
    [quartz_phonon_file],
    [quartz_phonon_file, '--btol=1000'],
    [nacl_phonon_file],
    [nacl_phonon_hdf5_file]]


@pytest.mark.integration
class TestRegression:

    @pytest.fixture
    def inject_mocks(self, mocker):
        # Prevent calls to show so we can get the current figure using
        # gcf()
        mocker.patch('matplotlib.pyplot.show')
        mocker.resetall()

    def teardown_method(self):
        # Ensure figures are closed
        matplotlib.pyplot.close('all')

    @pytest.mark.parametrize('dispersion_args', disp_params)
    def test_plots_produce_expected_xydata(
            self, inject_mocks, dispersion_args):
        euphonic.cli.dispersion.main(dispersion_args)

        lines = matplotlib.pyplot.gcf().axes[0].lines

        with open(disp_output_file, 'r') as f:
            expected_lines = json.load(f)[args_to_key(dispersion_args)]
        for index, line in enumerate(lines):
            npt.assert_allclose(
                line.get_xydata().T, np.array(expected_lines[index]),
                atol=sys.float_info.epsilon)

    @pytest.mark.parametrize('dispersion_args', [
        [quartz_phonon_file, '--save-to'],
        [quartz_phonon_file, '-s']])
    def test_plot_save_to_file(self, inject_mocks, tmpdir, dispersion_args):
        output_file = str(tmpdir.join('test.png'))
        euphonic.cli.dispersion.main(dispersion_args + [output_file])
        assert os.path.exists(output_file)


@patch('matplotlib.pyplot.show')
@pytest.mark.skip(reason='Only run if you want to regenerate the test data')
def test_regenerate_disp_data(_):
    json_data = {}
    for disp_param in disp_params:
        # Generate current figure for us to retrieve with gcf
        euphonic.cli.dispersion.main(disp_param)
        # Retrieve with gcf and record data
        json_data[args_to_key(disp_param)] = get_current_plot_lines_xydata()
    with open(disp_output_file, 'w+') as json_file:
        json.dump(json_data, json_file, indent=4)
