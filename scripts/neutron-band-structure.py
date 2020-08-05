#! /usr/bin/env python3
import argparse
import os
from typing import List, Tuple

import euphonic
from euphonic import ureg
import euphonic.plot
import matplotlib.pyplot as plt
import numpy as np
import seekpath


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)
    parser.add_argument('--ebins', type=int, default=200)
    parser.add_argument('--seekpath-labels', action='store_true',
                        dest='seekpath_labels',
                        help='Use the exact labels reported by Seekpath when '
                             'constructing the band structure path. Otherwise,'
                             ' use the labels detected by Euphonic.')
    parser.add_argument('--q-distance', type=float, default=0.025,
                        dest='q_distance',
                        help='Target distance between q-point samples')
    parser.add_argument('--x-label', type=str, default=None,
                        dest='x_label')
    parser.add_argument('--y-label', type=str, default=None,
                        dest='y_label')
    parser.add_argument('--title', type=str, default=None)
    parser.add_argument('--cmap', type=str, default='viridis',
                        help='Matplotlib colormap')
    parser.add_argument('--q-broadening', type=float, default=None,
                        dest='gaussian_x',
                        help='Width of Gaussian broadening on q axis in recip '
                             'angstrom. (No broadening if unspecified.)')
    parser.add_argument('--energy-broadening', type=float, default=None,
                        dest='gaussian_y',
                        help='Width of Gaussian broadening on energy axis in '
                        'meV. (No broadening if unspecified.)')
    return parser


def get_seekpath_structure(crystal: euphonic.Crystal) -> Tuple[np.ndarray,
                                                               np.ndarray,
                                                               List[int]]:
    # Seekpath needs a set of integer identities, while Crystal stores strings
    # so we need to convert e.g. ['H', 'C', 'Cl', 'C'] -> [0, 1, 2, 1]
    all_atom_types = list(set(crystal.atom_type.tolist()))
    atom_id_list = [all_atom_types.index(symbol)
                    for symbol in crystal.atom_type]

    return (crystal.cell_vectors.to('angstrom').magnitude,
            crystal.atom_r,
            atom_id_list)


def _get_break_points(bandpath: dict) -> Tuple[List[Tuple[int, int]],
                                               List[int]]:
    """Get information about band path labels and break points

    Parameters
    ----------
    bandpath
        Bandpath dictionary from Seekpath

    Returns
    -------
    (break_points, special_point_indices)

    ``break_points`` is a list of tuples identifying the boundaries of regions
    with discontinuities between them, e.g.::

      [(0, end1), (start2, end2), (start3, end3)]

    for a band path with three segments. end3 should equal the index of the
    last k-point in the full path.

    ``special_point_indices`` is a list of locations for labelled points in the
    full set of labels from Seekpath.

    """
    # Find break points between continuous spectra: wherever there are two
    # adjacent labels
    special_point_bools = np.fromiter(
        map(bool, bandpath["explicit_kpoints_labels"]), dtype=bool)
    special_point_indices = special_point_bools.nonzero()[0]

    # [T F F T T F T] -> [F F T T F T] AND [T F F T T F] = [F F F T F F] -> 3,
    break_points = (np.logical_and(special_point_bools[:-1],
                                   special_point_bools[1:])
                    .nonzero()[0].tolist())

    return break_points, special_point_indices


def _get_tick_labels(region: Tuple[int, int],
                     bandpath: dict,
                     special_point_indices) -> List[Tuple[int, str]]:
    start, end = region
    label_indices = special_point_indices[
        np.logical_and(special_point_indices >= start,
                       special_point_indices < end)]

    labels = [bandpath["explicit_kpoints_labels"][i]
              for i in label_indices]
    for i, label in enumerate(labels):
        if label == 'GAMMA':
            labels[i] = r'$\Gamma$'

    label_indices_shifted = [i - start for i in label_indices]
    return list(zip(label_indices_shifted, labels))


def main():
    args = get_parser().parse_args()
    filename = args.file
    summary_name = os.path.basename(filename)
    path = os.path.dirname(filename)

    print(f"Reading force constants from {filename}")
    force_constants = euphonic.ForceConstants.from_phonopy(
        path=path, summary_name=summary_name)

    print(f"Getting band path")
    structure = get_seekpath_structure(force_constants.crystal)
    bandpath = seekpath.get_explicit_k_path(structure,
                                            reference_distance=args.q_distance)

    break_points, special_point_indices = _get_break_points(bandpath)
    if break_points:
        print(f"Found {len(break_points)} regions in q-point path")

    regions = []
    start_point = 0
    for break_point in break_points:
        regions.append((start_point, break_point + 1))
        start_point = break_point + 1
    regions.append((start_point, len(bandpath["explicit_kpoints_rel"])))

    print("Computing phonon modes")
    region_modes = []
    for (start, end) in regions:
        qpts = bandpath["explicit_kpoints_rel"][start:end]
        region_modes.append(force_constants
                            .calculate_qpoint_phonon_modes(qpts,
                                                           reduce_qpts=False))

    frequency_units = region_modes[0].frequencies.units
    emin = min(np.min(modes.frequencies.magnitude)
               for modes in region_modes) * frequency_units
    emax = max(np.max(modes.frequencies.magnitude)
               for modes in region_modes) * frequency_units
    ebins = np.linspace(emin.to('meV').magnitude, emax.to('meV').magnitude,
                        args.ebins) * ureg['meV']

    print("Computing structure factors and generating 2D maps")
    spectra = []
    for region, modes in zip(regions, region_modes):

        structure_factor = modes.calculate_structure_factor()
        sqw = structure_factor.calculate_sqw_map(ebins)

        if args.seekpath_labels:
            sqw.x_tick_labels = _get_tick_labels(region, bandpath,
                                                 special_point_indices)
        if args.gaussian_x or args.gaussian_y:
            sqw = sqw.broaden(x_width=(args.gaussian_x * ureg['1 / angstrom']
                                       if args.gaussian_x else None),
                              y_width=(args.gaussian_y * ureg['meV']
                                       if args.gaussian_y else None))

        spectra.append(sqw)

    print(f"Plotting figure")
    if args.y_label is None:
        y_label = f"Energy / {spectra[0].y_data.units:~P}"
    else:
        y_label = args.y_label

    euphonic.plot.plot_2d(spectra,
                          cmap=args.cmap,
                          x_label=args.x_label,
                          y_label=y_label,
                          title=args.title)

    plt.show()


if __name__ == '__main__':
    main()
