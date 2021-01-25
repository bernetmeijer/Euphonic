import numpy as np

from euphonic.readers import phonopy
from euphonic import ForceConstants

##### INSTRUCTIONS #####

# how to run:
# ipython gulp_reader.py control_file force_constants_file

# to add to for new material:

# dictionary of masses

masses={'H': 1.01, 'C': 12.01}



# read FORCE_CONSTANTS file
def extract_force_constants(fc_pathname, n_atoms, n_cells):
    """
    Reads force constants from a Phonopy FORCE_CONSTANTS file
    Parameters
    ----------
    fc_pathname : str
        The FORCE_CONSTANTS file to read from
    n_atoms : int
        Number of atoms in the unit cell
    n_cells : int
        Number of unit cells in the supercell
    cell_origins_map : (n_atoms*n_cells, 2) int ndarray
        In the case of of non-diagonal supercell_matrices, the cell
        origins are not the same for each atom. This is a map of the
        equivalent cell origins for each atom, which is required to
        reorder the force constants matrix so that all atoms in the unit
        cell share equivalent origins.
    sc_relative_idx : (n_cells, n_cells) int ndarray
        The index n of the equivalent vector in cell_origins for each
        cell_origins[i] -> cell_origins[j] vector in the supercell.
        See _get_supercell_relative_idx
    p2s_map : (n_atoms,) int ndarray
        The index of the primitive atoms in the atoms of the supercell.
        Used if fc_format = 'full'
    Returns
    -------
    fc : (n_cells, 3*n_atoms, 3*n_atoms) float ndarray
        The force constants, in Euphonic convention
    """
    with open(fc_pathname, 'r') as f:
        fc_dims =  [int(dim) for dim in f.readline().split()]
    # single shape specifier implies full format
    if len(fc_dims) == 1:
        fc_dims.append(fc_dims[0])
    #_check_fc_shape(fc_dims, n_atoms, n_cells, fc_pathname, summary_name)


    fc = np.genfromtxt(fc_pathname, skip_header=1,
                           max_rows=4*(n_atoms*n_cells)**2, usecols=(0,1,2),
                           invalid_raise=False)
    
    if fc_dims[0] == fc_dims[1]:  # full fc
        fc = fc.reshape(n_atoms*n_cells, n_atoms, n_cells, 3, 3)
        # now shape is (n_atom, n_atom, n_cell, 3, 3)
        #fc = fc[p2s_map]
        
    # shape we need for euphonic
    # fc_euphonic = np.full((n_atoms, n_cells, n_atoms, 3, 3), -1.0)

    fc = fc.reshape((n_atoms, n_cells, n_atoms, 3, 3))

    fc = np.reshape(np.transpose(fc,
                                 axes=[1, 0, 3, 2, 4]), (n_cells, 3*n_atoms, 3*n_atoms))
    
    
    return np.array(fc)



# read CONTROL file
def read_control(filename):
    
    # dictionary for the crystal
    Dict={}
    
    # main dictionary, to which to add the crystal dictionary
    main_Dict={}
    
    cell_vectors=[]
    atom_r=[]
    dielectric=[]
    born=[]
    
    with open(filename, 'r') as cf:
        for i, line in enumerate(cf):
            if 'nelements' in line:
                n_elements=line.split()[1]
                n_elements=np.float(n_elements.replace(',', ''))
                continue
                
            if 'natoms' in line:
                n_atoms=line.split()[1]
                n_atoms=np.int(n_atoms.replace(',', ''))
                Dict['n_atoms']=n_atoms
                continue
                
            if 'ngrid(:)=1 1 1' in line:
                n_cells_in_sc=1
                main_Dict['n_cells_in_sc']=n_cells_in_sc
                main_Dict['sc_matrix']=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                continue
                
            if 'lfactor=0.1' in line:
                Dict['cell_vectors_unit']='angstrom'
                continue
                
            if 'lattvec(:,1)' in line:
                lst=line.split()[1:4]
                cell_vectors.append([float(i) for i in lst])
                continue
                
            if 'lattvec(:,2)' in line:
                lst=line.split()[1:4]
                cell_vectors.append([float(i) for i in lst])
                continue
                
            if 'lattvec(:,3)' in line:
                lst=line.split()[1:4]
                cell_vectors.append([float(i) for i in lst])
                continue
                
            if 'elements' in line:
                elements=[]
                terms=line.split()
                el1=terms[0].replace('elements=','')
                el1=el1.replace('"', '')
                elements.append(el1)
                for term in terms[1:]:
                    el=term.replace('"', '')
                    elements.append(el)
                elements=np.array(elements)

            if 'types' in line:
                atom_type=[]
                for term in line.split()[1:-1]:
                    el=elements[np.int(term)-1]
                    atom_type.append(el)
                Dict['atom_type']=np.array(atom_type)
                continue
                
            if 'positions' in line:
                if ':,1' in line:
                    position=line.split()[1:4]
                else:
                    position=line.split()[2:5]
                atom_r.append([float(i) for i in position])
                continue
            
            if 'epsilon' in line:
                lst=line.split()[1:4]
                dielectric.append([float(i) for i in lst])
                continue
                
            if 'born' in line:
                terms=line.split()
                length=len(terms)
                lst=line.split()[length-4:length-1]
                born.append([float(i) for i in lst])
                continue
    
    main_Dict['dielectric']=np.array(dielectric)
    main_Dict['dielectric_unit']='elementary_charge ** 2 / bohr / hartree'
    born=np.array(born)
    main_Dict['born']=born.reshape((n_atoms, 3, 3))
    main_Dict['born_unit']='elementary_charge'
    Dict['cell_vectors']=np.array(cell_vectors)
    Dict['atom_r']=np.array(atom_r)
    Dict['atom_mass_unit']='unified_atomic_mass_unit'
    main_Dict['cell_origins']=np.array([[0, 0, 0]])
    
    # get atom mass
    atom_mass=[]
    for i in atom_type:
        atom_mass.append(masses[i])
    Dict['atom_mass']=np.array(atom_mass)
    
    main_Dict['crystal']=Dict
    
    return main_Dict


# combine everything
def read_gulp(CONTROL_file, FORCE_CONSTANTS_file):
    main_Dict=read_control(CONTROL_file)
    n_atoms=main_Dict['crystal']['n_atoms']
    n_cells=main_Dict['n_cells_in_sc']
    
    force_constants=extract_force_constants(FORCE_CONSTANTS_file, n_atoms, n_cells)
    
    main_Dict['force_constants']=force_constants
    main_Dict['force_constants_unit']='hartree / bohr ** 2'
    
    return ForceConstants.from_dict(main_Dict)
