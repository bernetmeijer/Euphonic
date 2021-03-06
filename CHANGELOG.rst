`Unreleased <https://github.com/pace-neutrons/Euphonic/compare/v0.3.2...HEAD>`_
----------

- Changes:

  - Some of Euphonic's dependency version requirements have been changed - but
    can now be relied on with more certainty due to better CI testing. This
    includes:

    - numpy requirement increased from ``1.9.1`` to ``1.12.1``
    - matplotlib requirement increased from ``1.4.2`` to ``2.0.0``
    - pint requirement decreased from ``0.10.1`` to ``0.9``
    - h5py requirement decreased from ``2.9.0`` to ``2.7.0``
    - pyyaml requirement decreased from ``5.1.2`` to ``3.13``

- Improvements:

  - ``yaml.CSafeLoader`` is now used instead of ``yaml.SafeLoader`` by
    default, so Phonopy ``.yaml`` files should load faster
  - Metadata ``__euphonic_version__`` and ``__euphonic_class__`` have been
    added to .json file output for better provenance

- Bug fixes:

  - Fix read of Phonopy 'full' force constants from phonopy.yaml and
    FORCE_CONSTANTS files
  - Fix structure factor calculation at gamma points with splitting, see
    `#107 <https://github.com/pace-neutrons/Euphonic/issues/107>`_
  - Change broadening implementation from ``scipy.signal.fftconvolve``
    to use ``scipy.ndimage`` functions for better handling of bright
    Bragg peaks, see
    `#108 <https://github.com/pace-neutrons/Euphonic/issues/108>`_

`v0.3.2 <https://github.com/pace-neutrons/Euphonic/compare/v0.3.1...v0.3.2>`_
----------

- New Features:

  - Added `weights` as an argument to
    `ForceConstants.calculate_qpoint_phonon_modes`, this will allow easier
    use of symmetry reduction for calculating density of states, for example.
  - Modules have been added to support spherical averaging from 3D
    q-points to mod(q)

    - euphonic.sampling provides pure functions for the generation of
      points on (2D) unit square and (3D) unit sphere surfaces.
    - A script is provided for visualisation of the different schemes
      implemented in euphonic.sampling. This is primarily intended for
      education and debugging.
    - euphonic.powder provides functions which, given force constants
      data, can use these sampling methods to obtain
      spherically-averaged phonon DOS and coherent structure factor
      data as 1D spectrum objects. (It is anticipated that this module
      will grow to include schemes beyond this average over a single
      sphere.)
  - Added ``Crystal.to_spglib_cell`` convenience function

- Changes:

  - The Scripts folder has been removed. Command-line tools are now
    located in the euphonic.cli module. The entry-points are managed
    in setup.py, and each tool has the prefix "euphonic-" to avoid
    namespace clashes with other tools on the user's
    computer. (e.g. euphonic-dos)
  - From an interactive shell with tab-completion, one can find all
    the euphonic tools by typing "euphonic-<TAB>".
  - Changed arguments for ``util.get_qpoint_labels(Crystal, qpts)``
    to ``util.get_qpoint_labels(qpts, cell=None)`` where
    ``cell = Crystal.to_spglib_cell()``

- Bug fixes:

  - Correctly convert from Phonopy's q-point weight convention to Euphonic's
    when reading from mesh.yaml (see
    `7509043 <https://github.com/pace-neutrons/Euphonic/commit/7509043>`_)
  - Avoid IndexError in ``ForceConstants.calculate_qpoint_phonon_modes`` when
    there is only one q-point (which is gamma) and ``splitting=True``

`v0.3.1 <https://github.com/pace-neutrons/Euphonic/compare/v0.3.0...v0.3.1>`_
----------

- New Features:

  - A system has been added for reference data in JSON files. These
    are accessed via ``euphonic.utils.get_reference_data`` and some
    data has been added for coherent scattering lengths and cross-sections.
    This system has been made available to the
    ``calculate_structure_factor()`` method; it is no longer necessary to
    craft a data dict every time a program uses this function.

- Changes:

  - Fixed structure factor formula in docs (``|F(Q, nu)|`` -> ``|F(Q, \\nu)|^2``
    and ``e^(Q.r)`` -> ``e^(iQ.r)``)

- Bug fixes:

  - Fix ``'born':null`` in ``ForceConstants`` .json files when Born is not
    present in the calculation (see
    `c20679c <https://github.com/pace-neutrons/Euphonic/commit/c20679c>`_)
  - Fix incorrect calculation of LO-TO splitting when ``reduce_qpts=True``,
    as the 'reduced' q rather than the actual q was used as the q-direction
    (see `3958072 <https://github.com/pace-neutrons/Euphonic/commit/3958072>`_)
  - Fix interpolation for materials with non-symmetric supcercell matrices,
    see `#81 <https://github.com/pace-neutrons/Euphonic/issues/81>`_
  - Fix interpolation for force constants read from Phonopy for materials that
    have a primitive matrix and more than 1 species, see
    `#77 <https://github.com/pace-neutrons/Euphonic/issues/77>`_

`v0.3.0 <https://github.com/pace-neutrons/Euphonic/compare/v0.2.2...v0.3.0>`_
----------

- Breaking Changes:

  - There has been a major refactor, for see the latest
    `docs <https://euphonic.readthedocs.io/en/latest>`_ for how to use, or
    `here <https://euphonic.readthedocs.io/en/latest/refactor.html>`_ for
    refactor details
  - Python 2 is no longer supported. Supported Python versions are ``3.6``,
    ``3.7`` and ``3.8``

- New Features:

  - Euphonic can now read Phonopy input! See
    `the docs <https://euphonic.readthedocs.io/en/latest>`_
    for details.

- Improvements:

  - Added ``fall_back_on_python`` boolean keyword argument to
    ``ForceConstants.calculate_qpoint_phonon_modes`` to control
    whether the Python implementation is used as a fallback to the C
    extension or not, see
    `#35 <https://github.com/pace-neutrons/Euphonic/issues/35>`_
  - Added ``--python-only`` option to ``setup.py`` to enable install
    without the C extension

- Bug fixes:

  - On reading CASTEP phonon file header information, switch from a fixed
    number of lines skipped to a search for a specific line, fixing issue
    `#23 <https://github.com/pace-neutrons/Euphonic/issues/23>`_
  - Fix NaN frequencies/eigenvectors for consecutive gamma points, see
    `#25 <https://github.com/pace-neutrons/Euphonic/issues/25>`_
  - Fix issue saving plots to file with dispersion.py, see
    `#27 <https://github.com/pace-neutrons/Euphonic/issues/27>`_
  - Fix incorrect frequencies at gamma point when using dipole correction
    in C, `#45 <https://github.com/pace-neutrons/Euphonic/issues/45>`_

`v0.2.2 <https://github.com/pace-neutrons/Euphonic/compare/v0.2.1...v0.2.2>`_
------

- Bug fixes:

  - Add MANIFEST.in for PyPI distribution

`v0.2.1 <https://github.com/pace-neutrons/Euphonic/compare/v0.2.0...v0.2.1>`_
------

- Bug fixes:

  - Cannot easily upload C header files to PyPI without an accompanying source
    file, so refactor C files to avoid this

`v0.2.0 <https://github.com/pace-neutrons/Euphonic/compare/v0.1-dev3...v0.2.0>`_
------

- There are several breaking changes:

  - Changes to the object instantiation API. The former interface
    ``InterpolationData(seedname)`` has been changed to
    ``InterpolationData.from_castep(seedname)`,` in anticipation of more codes
    being added which require more varied arguments.
  - Changes to the Debye-Waller calculation API when calculating the structure
    factor. The previous ``dw_arg`` kwarg accepted either a seedname or length
    3 list describing a grid. The new kwarg is now ``dw_data`` and accepts a
    ``PhononData`` or ``InterpolationData`` object with the frequencies
    calculated on a grid. This is to make it clearer to the user exactly what
    arguments are being used when calculating phonons on the grid.
  - Changes to parallel functionality. The previous parallel implementation
    based on Python's multiprocessing has been removed and replaced by a
    C/OpenMP version. This has both better performance and is more robust. As
    a result the ``n_procs`` kwarg to ``calculate_fine_phonons`` has been
    replaced by ``use_c`` and ``n_threads`` kwargs.

- Improvements:

  - The parallel implementation based on Python's multiprocessing has been
    removed and now uses C/OpenMP which both has better performance and is more
    robust
  - Documentation has been moved to readthedocs and is more detailed
  - Clearer interface for calculating the Debye-Waller factor
  - Better error handling (e.g. empty ``InterpolationData`` objects, Matplotlib
    is not installed...)

- Bug fixes:

  - Fix gwidth for DOS not being converted to correct units
  - Fix qwidth for S(Q,w) broadening being incorrectly calculated
