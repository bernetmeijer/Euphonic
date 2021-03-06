import json
from unittest.mock import patch
import euphonic.cli.dispersion
from tests_and_analysis.test.script_tests.utils import (
    get_phonon_file, get_dispersion_params, get_current_plot_lines_xydata,
    get_dispersion_data_file)


@patch("matplotlib.pyplot.show")
def regenerate_dispersion_data(_):

    json_data = {}

    for dispersion_params in get_dispersion_params():
        # Generate current figure for us to retrieve with gcf
        euphonic.cli.dispersion.main([get_phonon_file()] + dispersion_params)

        # Retrieve with gcf and write to file
        json_data[" ".join(
            dispersion_params)] = get_current_plot_lines_xydata()

    with open(get_dispersion_data_file(), "w+") as json_file:
        json.dump(json_data, json_file, indent=4)


if __name__ == "__main__":
    regenerate_dispersion_data()
