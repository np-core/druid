"""
Code adopted from the MaSIF protein surface prediction model:
https://www.biorxiv.org/content/10.1101/606202v1

Pablo Gainza - LPDI STI EPFL 2019
This file is part of MaSIF.

Released under an Apache License 2.0
"""

import os
import logging
import numpy as np
import pyvista as pv
from pathlib import Path
from subprocess import PIPE, Popen
import pymesh

from Bio.PDB import *
from Bio.PDB import StructureBuilder
from Bio.SeqUtils import IUPACData

from druid.utils import PoreLogger


# Kyte Doolittle scale: https://en.wikipedia.org/wiki/Hydrophobicity_scales
KD_SCALE = dict()
KD_SCALE["ILE"] = 4.5
KD_SCALE["VAL"] = 4.2
KD_SCALE["LEU"] = 3.8
KD_SCALE["PHE"] = 2.8
KD_SCALE["CYS"] = 2.5
KD_SCALE["MET"] = 1.9
KD_SCALE["ALA"] = 1.8
KD_SCALE["GLY"] = -0.4
KD_SCALE["THR"] = -0.7
KD_SCALE["SER"] = -0.8
KD_SCALE["TRP"] = -0.9
KD_SCALE["TYR"] = -1.3
KD_SCALE["PRO"] = -1.6
KD_SCALE["HIS"] = -3.2
KD_SCALE["GLU"] = -3.5
KD_SCALE["GLN"] = -3.5
KD_SCALE["ASP"] = -3.5
KD_SCALE["ASN"] = -3.5
KD_SCALE["LYS"] = -3.9
KD_SCALE["ARG"] = -4.5


class ProteinModel(PoreLogger):

    """ ProteinModel from PDB for feature extraction and computation """

    def __init__ (self, pdb_id: str, outdir: Path, chains: str):

        PoreLogger.__init__(self, name="ProteinModel", level=logging.INFO)

        self.outdir = outdir  # the working output directory
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.pdb_id = pdb_id  # the pdb id
        self.chains = chains  # the chains to use
        self.pdb_file = self.download()  # the downloaded pdb file

        self.surface_model = dict()  # the protein surface model from MSMS

        # Data preparation and triangulation pipeline
        self.proton_pdb_file = Path(f"{self.outdir / f'{self.pdb_id}.proton.pdb'}")  # the protonated pdb file
        self.chains_pdb_file = Path(f"{self.outdir / f'{self.pdb_id}.chains.pdb'}")  # the extracted chains
        self.chains_pdb_file = Path(f"{self.outdir / f'{self.pdb_id}.chains.pdb'}")  # the extracted chains

    def prepare_feature_data(self, compute_hbond: bool = True, compute_hphob: bool = True, mesh_resolution: float = 1.0):

        """ Data preparation and feature extraction for MaSIF """

        # Triangulate surface and chemical features
        tricorder = Tricorder(protein_model=self)

        # Protonate the file (strip before protonation)
        proton_pdb_file = tricorder.protonate(
            pdb_file=self.pdb_file,
            pdb_out=self.proton_pdb_file
        )

        # Extract the chains from the PDB
        tricorder.extract_pdb(
            pdb_file=proton_pdb_file,
            pdb_out=self.chains_pdb_file,
            extract_chains=self.chains
        )

        # Compute MSMS of surface w/ hydrogens and parameters from paper
        self.surface_model = tricorder.compute_surface_mesh(density=3.0, hdensity=3.0, probe_radius=1.5)

        # self.visualize_surface_mesh(
        #     vertices=self.surface_model['vertices'], faces=self.surface_model['faces']
        # )

        if compute_hbond:
            # Compute "charged" vertices
            vertex_hbond = tricorder.compute_charges(
                self.surface_model['vertices'], self.surface_model['names']
            )

        if compute_hphob:
            # For each surface residue, assign the hydrophobicity of its amino acid
            vertex_hphobicity = tricorder.compute_hydrophobicity(self.surface_model['names'])

        # Fix the mesh and regularize (see paper)
        mesh = pymesh(
            self.surface_model['vertices'], self.surface_model['faces']
        )

        print(mesh)

        # regular_mesh = fix_mesh(mesh, 1.0)

    @staticmethod
    def visualize_surface_mesh(vertices, faces):

        surf = pv.PolyData(vertices, faces)
        surf.show()

    def download(self):

        self.logger.info(f"Downloading: {self.pdb_id} from PDB")
        pdbl = PDBList(server='http://ftp.wwpdb.org')
        self.pdb_file = Path(
            pdbl.retrieve_pdb_file(self.pdb_id, pdir=str(self.outdir), file_format='pdb')
        )
        self.logger.info(f"Download complete file at: {self.pdb_file }")

        return self.pdb_file


# Utility toolkits


class Tricorder(PoreLogger):

    """ Class to extract structural and chemical features from a ProteinModel """

    def __init__(self, protein_model: ProteinModel):

        PoreLogger.__init__(self, name="Tricorder", level=logging.INFO)

        self.protein_model = protein_model  # the protein model to extract features from

        self.msms = "msms" # the MSMS surface prediction software binary

        self.protein_letters = [x.upper() for x in IUPACData.protein_letters_3to1.keys()]

    def protonate(self, pdb_file: Path, pdb_out: Path):

        """ Protonate (add hydrogens to) the protein using `reduce` """

        args = ["reduce", "-Trim", str(pdb_file)]

        self.logger.info(f"Removing protons from: {pdb_file}")
        p2 = Popen(args, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p2.communicate()
        with pdb_out.open('w') as outfile:
            outfile.write(stdout.decode('utf-8').rstrip())

        self.logger.info(f"Protonating structure: {self.protein_model.pdb_id} ")
        args = ["reduce", "-HIS", str(pdb_out)]
        p2 = Popen(args, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p2.communicate()
        with pdb_out.open('w') as outfile:
            outfile.write(stdout.decode('utf-8').rstrip())

        self.logger.info(f"Protonation complete: {pdb_out}")

        return pdb_out

    def find_modified_amino_acids(self, path: Path):

        """ Find modified amino acids in the PDB (e.g. MSE)

        Contributed by github user jomimc
        Pablo Gainza - LPDI STI EPFL 2019
        Released under an Apache License 2.0
        """

        res_set = set()
        for line in path.open():
            if line[:6] == 'SEQRES':
                for res in line.split()[4:]:
                    res_set.add(res)
        for res in list(res_set):
            if res in self.protein_letters:
                res_set.remove(res)
        return res_set

    def extract_pdb(self, pdb_file: Path, pdb_out: Path, extract_chains: str):

        """ Extract selected chains from a PDB and save in new file

        Pablo Gainza - LPDI STI EPFL 2019
        Released under an Apache License 2.0
        """

        parser = PDBParser(QUIET=True)
        struct = parser.get_structure(str(pdb_file), str(pdb_file))
        model = Selection.unfold_entities(struct, "M")[0]

        # _chains = Selection.unfold_entities(struct, "C")

        # Select residues to extract and build new structure
        struct_build = StructureBuilder.StructureBuilder()
        struct_build.init_structure("output")
        struct_build.init_seg(" ")
        struct_build.init_model(0)
        output_struct = struct_build.get_structure()

        # Load a list of non-standard amino acid names -- these are
        # typically listed under HETATM, so they would be typically
        # ignored by the original algorithm
        modified_amino_acids = self.find_modified_amino_acids(path=pdb_file)

        for chain in model:
            if extract_chains or chain.get_id() in extract_chains:
                struct_build.init_chain(chain.get_id())
                for residue in chain:
                    het = residue.get_id()
                    if het[0] == " ":
                        output_struct[0][chain.get_id()].add(residue)
                    elif het[0][-3:] in modified_amino_acids:
                        output_struct[0][chain.get_id()].add(residue)

        # Output the selected residues
        pdbio = PDBIO()
        pdbio.set_structure(output_struct)
        pdbio.save(str(pdb_out), select=NotDisordered())

    def compute_surface_mesh(self, density: float = 3.0, hdensity: float = 3.0, probe_radius: float = 1.5):

        """ Calls MSMS and returns the vertices.
        Special atoms are atoms with a reduced radius.

        Pablo Gainza LPDI EPFL 2017-2019

        :param density: surface points density
        :param hdensity: surface points high density
        :param probe_radius:
        """

        msms_base = str((self.protein_model.outdir / f"{self.protein_model.pdb_id}").absolute())

        self.write_chain_spheres(self.protein_model.chains_pdb_file, msms_base+".xyzrn")

        args = [self.msms, "-density", f"{density}", "-hdensity", f"{hdensity}", "-probe_radius",
                f"{probe_radius}", "-if", msms_base+".xyzrn", "-of", msms_base, "-af", msms_base]

        p2 = Popen(args, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p2.communicate()

        # Extract surface graph from MSMS files
        vertices, faces, normals, names = self.read_msms(file_root=msms_base)

        # Extract areas from file
        self.logger.info("Extracting surface area data from MSMS output")
        areas = {}
        ses_file = open(msms_base + ".area")
        next(ses_file)  # ignore header line
        for line in ses_file:
            fields = line.split()
            areas[fields[3]] = fields[1]
        ses_file.close()

        # Remove temporary files.
        for ext in ('.area', '.xyzrn', '.vert', '.face'):
            os.remove(msms_base + ext)

        self.logger.info("Completed MSMS surface prediction and data extraction")
        self.logger.info(
            f"Surface mesh for protein ({self.protein_model.pdb_id}): "
            f"{len(vertices)} vertices {len(faces)} faces {len(areas)} areas"
        )

        return {
            'vertices': vertices, 'faces': faces, 'normals': normals, 'names': names, 'areas': areas
        }

    def write_chain_spheres(self, chain_file: Path, xyzrn_out: str):

        self.logger.info(f"Writing chain spheres for MSMS surface prediction: {chain_file}")

        parser = PDBParser()
        struct = parser.get_structure(chain_file, chain_file)
        outfile = open(xyzrn_out, "w")
        for atom in struct.get_atoms():
            name = atom.get_name()
            residue = atom.get_parent()
            # Ignore HETATMS
            if residue.get_id()[0] != " ":
                continue
            resname = residue.get_resname()
            # reskey = residue.get_id()[1]  # was not used by script
            chain = residue.get_parent().get_id()
            atomtype = name[0]

            color = "Green"
            coords = None
            # Only the following block is written (coords != None)
            if atomtype in RADII and resname in PolarHydrogens:
                if atomtype == "O":
                    color = "Red"
                if atomtype == "N":
                    color = "Blue"
                if atomtype == "H":
                    if name in PolarHydrogens[resname]:
                        color = "Blue"  # Polar hydrogens
                coords = "{:.06f} {:.06f} {:.06f}".format(
                    atom.get_coord()[0], atom.get_coord()[1], atom.get_coord()[2]
                )

            # es: sanity check if same output moved insertion and full_id <-- tab-left

            insertion = "x"
            if residue.get_id()[2] != " ":
                insertion = residue.get_id()[2]

            full_id = "{}_{:d}_{}_{}_{}_{}".format(
                chain, residue.get_id()[1], insertion, resname, name, color
            )

            if coords is not None:
                outfile.write(coords + " " + RADII[atomtype] + " 1 " + full_id + "\n")

        outfile.close()
        self.logger.info(f"Chain spheres for MSMS written to: {xyzrn_out}")

    def read_msms(self, file_root: str):

        """ Read the surface from the MSMS output """

        vert_file = Path(file_root + ".vert")
        self.logger.info(f"Parsing MSMS mesh vertices: {vert_file}")
        with vert_file.open() as vertfile:
            meshdata = (vertfile.read().rstrip()).split("\n")

        # Read number of vertices.
        count = {}
        header = meshdata[2].split()
        count["vertices"] = int(header[0])

        self.logger.info(f"Creating data structures from vertices: {vert_file}")

        # Data Structures (see docstring)
        vertices = np.zeros((count["vertices"], 3))
        normalv = np.zeros((count["vertices"], 3))
        atom_id = [""] * count["vertices"]
        res_id = [""] * count["vertices"]
        for i in range(3, len(meshdata)):
            fields = meshdata[i].split()
            vi = i - 3
            vertices[vi][0] = float(fields[0])
            vertices[vi][1] = float(fields[1])
            vertices[vi][2] = float(fields[2])
            normalv[vi][0] = float(fields[3])
            normalv[vi][1] = float(fields[4])
            normalv[vi][2] = float(fields[5])
            atom_id[vi] = fields[7]
            res_id[vi] = fields[9]
            # sanity counter
            count["vertices"] -= 1

        # Read faces.
        face_file = Path(file_root + ".face")
        self.logger.info(f"Parsing MSMS mesh faces: {face_file}")
        with face_file.open() as facefile:
            face_meshdata = (facefile.read().rstrip()).split("\n")

        # Read number of vertices.
        header = face_meshdata[2].split()
        count["faces"] = int(header[0])
        faces = np.zeros((count["faces"], 3), dtype=int)

        for i in range(3, len(face_meshdata)):
            fi = i - 3
            fields = face_meshdata[i].split()

            try:
                faces[fi][0] = int(fields[0]) - 1
            except IndexError:
                continue

            try:
                faces[fi][1] = int(fields[1]) - 1
            except IndexError:
                continue

            try:
                faces[fi][2] = int(fields[2]) - 1
            except IndexError:
                continue

            count["faces"] -= 1

        # sanity checks will fail if IndexError and continued in last loop
        assert count["vertices"] == 0
        assert count["faces"] == 0

        self.logger.info(f"Created mesh data from MSMS surface predictions")

        return vertices, faces, normalv, res_id

    def compute_hydrophobicity(self, names):

        self.logger.info("Computing hydrophobicity for each vertex in surface mesh [Kyte Doolittle scale]")

        hp = np.zeros(len(names))
        for ix, name in enumerate(names):
            aa = name.split("_")[3]
            hp[ix] = KD_SCALE[aa]

        return hp

    def compute_charges(self, vertices, names):

        # Compute vertex charges based on hydrogen bond potential.
        # pdb_filename: The filename of the protonated protein.
        # vertices: The surface vertices of the protonated protein
        # The name of each vertex in the format, example: B_125_x_ASN_ND2_Green
        # where B is chain, 125 res id, x the insertion, ASN aatype, ND2 the name of the
        # atom, and green is not used anymore.

        self.logger.info("Computing charges for each vertex in surface mesh based on hydrogen potential")

        parser = PDBParser(QUIET=True)
        struct = parser.get_structure(self.protein_model.chains_pdb_file.name, self.protein_model.chains_pdb_file)
        residues = {}
        for res in struct.get_residues():
            chain_id = res.get_parent().get_id()
            if chain_id == "":
                chain_id = " "
            residues[(chain_id, res.get_id())] = res

        atoms = Selection.unfold_entities(struct, "A")
        satisfied_CO, satisfied_HN = self._compute_satisfied_CO_HN(atoms)

        charge = np.array([0.0] * len(vertices))
        # Go over every vertex
        for ix, name in enumerate(names):
            fields = name.split("_")
            chain_id = fields[0]
            if chain_id == "":
                chain_id = " "
            if fields[2] == "x":
                fields[2] = " "
            res_id = (" ", int(fields[1]), fields[2])
            aa = fields[3]
            atom_name = fields[4]
            # Ignore atom if it is BB and it is already satisfied.
            if atom_name == "H" and res_id in satisfied_HN:
                continue
            if atom_name == "O" and res_id in satisfied_CO:
                continue
            # Compute the charge of the vertex
            charge[ix] = self._compute_charge_helper(
                atom_name, residues[(chain_id, res_id)], vertices[ix]
            )

        return charge

    # Compute the charge of a vertex in a residue.
    def _compute_charge_helper(self, atom_name, res, v):
        res_type = res.get_resname()
        # Check if it is a polar hydrogen.
        if self._is_polar_hydrogen(atom_name, res):
            donor_atom_name = DonorAtom[atom_name]
            a = res[donor_atom_name].get_coord()  # N/O
            b = res[atom_name].get_coord()  # H
            # Donor-H is always 180.0 degrees, = pi
            angle_deviation = self._compute_angle_deviation(a, b, v, np.pi)
            angle_penalty = self._compute_angle_penalty(angle_deviation)
            return 1.0 * angle_penalty
        # Check if it is an acceptor oxygen or nitrogen
        elif self._is_acceptor_atom(atom_name, res):
            acceptor_atom = res[atom_name]
            b = acceptor_atom.get_coord()
            try:
                a = res[AcceptorAngleAtom[atom_name]].get_coord()
            except:
                return 0.0
            # 120 degrees for acceptor
            angle_deviation = self._compute_angle_deviation(a, b, v, 2 * np.pi / 3)
            # TODO: This should not be 120 for all atoms, i.e. for HIS it should be ~125.0
            angle_penalty = self._compute_angle_penalty(angle_deviation)
            plane_penalty = 1.0
            if atom_name in acceptorPlaneAtom:
                try:
                    d = res[acceptorPlaneAtom[atom_name]].get_coord()
                except:
                    return 0.0
                plane_deviation = self._compute_plane_deviation(d, a, b, v)
                plane_penalty = self._compute_angle_penalty(plane_deviation)
            return -1.0 * angle_penalty * plane_penalty
            # Compute the
        return 0.0

    @staticmethod
    def _is_polar_hydrogen(atom_name, res):

        if atom_name in PolarHydrogens[res.get_resname()]:
            return True
        else:
            return False

    @staticmethod
    def _is_acceptor_atom(atom_name, res):
        if atom_name.startswith("O"):
            return True
        else:
            if res.get_resname() == "HIS":
                if atom_name == "ND1" and "HD1" not in res:
                    return True
                if atom_name == "NE2" and "HE2" not in res:
                    return True
        return False

    def _compute_satisfied_CO_HN(self, atoms):

        """ Compute the list of backbone C=O:H-N that are satisfied. These will be ignored. """

        ns = NeighborSearch(atoms)
        satisfied_CO = set()
        satisfied_HN = set()
        for atom1 in atoms:
            res1 = atom1.get_parent()
            if atom1.get_id() == "O":
                neigh_atoms = ns.search(atom1.get_coord(), 2.5, level="A")
                for atom2 in neigh_atoms:
                    if atom2.get_id() == "H":
                        res2 = atom2.get_parent()
                        # Ensure they belong to different residues.
                        if res2.get_id() != res1.get_id():
                            # Compute the angle N-H:O, ideal value is 180 (but in
                            # helices it is typically 160) 180 +-30 = pi
                            angle_N_H_O_dev = self._compute_angle_deviation(
                                res2["N"].get_coord(),
                                atom2.get_coord(),
                                atom1.get_coord(),
                                np.pi,
                            )
                            # Compute angle H:O=C, ideal value is ~160 +- 20 = 8*pi/9
                            angle_H_O_C_dev = self._compute_angle_deviation(
                                atom2.get_coord(),
                                atom1.get_coord(),
                                res1["C"].get_coord(),
                                8 * np.pi / 9,
                            )
                            # Allowed deviations: 30 degrees (pi/6) and 20 degrees (pi/9)
                            if (
                                angle_N_H_O_dev - np.pi / 6 < 0
                                and angle_H_O_C_dev - np.pi / 9 < 0.0
                            ):
                                satisfied_CO.add(res1.get_id())
                                satisfied_HN.add(res2.get_id())

        return satisfied_CO, satisfied_HN

    @staticmethod
    def _compute_angle_penalty(angle_deviation):
        """ angle_deviation from ideal value. TODO: do a more data-based solution """
        # Standard deviation: hbond_std_dev
        return max(0.0, 1.0 - (angle_deviation / (hbond_std_dev)) ** 2)

    @staticmethod
    def _compute_angle_deviation(a, b, c, theta):
        return abs(calc_angle(Vector(a), Vector(b), Vector(c)) - theta)

    @staticmethod
    def _compute_plane_deviation(a, b, c, d):
        """ Compute the angle deviation from a plane """
        dih = calc_dihedral(Vector(a), Vector(b), Vector(c), Vector(d))
        dev1 = abs(dih)
        dev2 = np.pi - abs(dih)
        return min(dev1, dev2)

# Exclude disordered atoms.
class NotDisordered(Select):
    def accept_atom(self, atom):
        return not atom.is_disordered() or atom.get_altloc() == "A" or atom.get_altloc() == "1"


# Chemical parameters for MaSIF.
# Pablo Gainza - LPDI STI EPFL 2018-2019
# Released under an Apache License 2.0

# radii for atoms in explicit case.
RADII = dict()
RADII["N"] = "1.540000"
RADII["N"] = "1.540000"
RADII["O"] = "1.400000"
RADII["C"] = "1.740000"
RADII["H"] = "1.200000"
RADII["S"] = "1.800000"
RADII["P"] = "1.800000"
RADII["Z"] = "1.39"
RADII["X"] = "0.770000"   # Radii of CB or CA in disembodied case.


# Polar hydrogen's names correspond to that of the program Reduce.
PolarHydrogens = dict()
PolarHydrogens["ALA"] = ["H"]
PolarHydrogens["GLY"] = ["H"]
PolarHydrogens["SER"] = ["H", "HG"]
PolarHydrogens["THR"] = ["H", "HG1"]
PolarHydrogens["LEU"] = ["H"]
PolarHydrogens["ILE"] = ["H"]
PolarHydrogens["VAL"] = ["H"]
PolarHydrogens["ASN"] = ["H", "HD21", "HD22"]
PolarHydrogens["GLN"] = ["H", "HE21", "HE22"]
PolarHydrogens["ARG"] = ["H", "HH11", "HH12", "HH21", "HH22", "HE"]
PolarHydrogens["HIS"] = ["H", "HD1", "HE2"]
PolarHydrogens["TRP"] = ["H", "HE1"]
PolarHydrogens["PHE"] = ["H"]
PolarHydrogens["TYR"] = ["H", "HH"]
PolarHydrogens["GLU"] = ["H"]
PolarHydrogens["ASP"] = ["H"]
PolarHydrogens["LYS"] = ["H", "HZ1", "HZ2", "HZ3"]
PolarHydrogens["PRO"] = []
PolarHydrogens["CYS"] = ["H"]
PolarHydrogens["MET"] = ["H"]

hbond_std_dev = np.pi / 3

# Dictionary from an acceptor atom to its directly bonded atom on which to
# compute the angle.
AcceptorAngleAtom = dict()
AcceptorAngleAtom["O"] = "C"
AcceptorAngleAtom["O1"] = "C"
AcceptorAngleAtom["O2"] = "C"
AcceptorAngleAtom["OXT"] = "C"
AcceptorAngleAtom["OT1"] = "C"
AcceptorAngleAtom["OT2"] = "C"

# Dictionary from acceptor atom to a third atom on which to compute the plane.
acceptorPlaneAtom = dict()
acceptorPlaneAtom["O"] = "CA"

# Dictionary from an H atom to its donor atom.
DonorAtom = dict()
DonorAtom["H"] = "N"

# Hydrogen bond information.
# ARG
# ARG NHX
# Angle: NH1, HH1X, point and NH2, HH2X, point 180 degrees.
# radii from HH: radii[H]
# ARG NE
# Angle: ~ 120 NE, HE, point, 180 degrees
DonorAtom["HH11"] = "NH1"
DonorAtom["HH12"] = "NH1"
DonorAtom["HH21"] = "NH2"
DonorAtom["HH22"] = "NH2"
DonorAtom["HE"] = "NE"

# ASN
# Angle ND2,HD2X: 180
# Plane: CG,ND2,OD1
# Angle CG-OD1-X: 120
DonorAtom["HD21"] = "ND2"
DonorAtom["HD22"] = "ND2"
# ASN Acceptor
AcceptorAngleAtom["OD1"] = "CG"
acceptorPlaneAtom["OD1"] = "CB"

# ASP
# Plane: CB-CG-OD1
# Angle CG-ODX-point: 120
AcceptorAngleAtom["OD2"] = "CG"
acceptorPlaneAtom["OD2"] = "CB"

# GLU
# PLANE: CD-OE1-OE2
# ANGLE: CD-OEX: 120
# GLN
# PLANE: CD-OE1-NE2
# Angle NE2,HE2X: 180
# ANGLE: CD-OE1: 120
DonorAtom["HE21"] = "NE2"
DonorAtom["HE22"] = "NE2"
AcceptorAngleAtom["OE1"] = "CD"
AcceptorAngleAtom["OE2"] = "CD"
acceptorPlaneAtom["OE1"] = "CG"
acceptorPlaneAtom["OE2"] = "CG"

# HIS Acceptors: ND1, NE2
# Plane ND1-CE1-NE2
# Angle: ND1-CE1 : 125.5
# Angle: NE2-CE1 : 125.5
AcceptorAngleAtom["ND1"] = "CE1"
AcceptorAngleAtom["NE2"] = "CE1"
acceptorPlaneAtom["ND1"] = "NE2"
acceptorPlaneAtom["NE2"] = "ND1"

# HIS Donors: ND1, NE2
# Angle ND1-HD1 : 180
# Angle NE2-HE2 : 180
DonorAtom["HD1"] = "ND1"
DonorAtom["HE2"] = "NE2"

# TRP Donor: NE1-HE1
# Angle NE1-HE1 : 180
DonorAtom["HE1"] = "NE1"

# LYS Donor NZ-HZX
# Angle NZ-HZX : 180
DonorAtom["HZ1"] = "NZ"
DonorAtom["HZ2"] = "NZ"
DonorAtom["HZ3"] = "NZ"

# TYR acceptor OH
# Plane: CE1-CZ-OH
# Angle: CZ-OH 120
AcceptorAngleAtom["OH"] = "CZ"
acceptorPlaneAtom["OH"] = "CE1"

# TYR donor: OH-HH
# Angle: OH-HH 180
DonorAtom["HH"] = "OH"
acceptorPlaneAtom["OH"] = "CE1"

# SER acceptor:
# Angle CB-OG-X: 120
AcceptorAngleAtom["OG"] = "CB"

# SER donor:
# Angle: OG-HG-X: 180
DonorAtom["HG"] = "OG"

# THR acceptor:
# Angle: CB-OG1-X: 120
AcceptorAngleAtom["OG1"] = "CB"

# THR donor:
# Angle: OG1-HG1-X: 180
DonorAtom["HG1"] = "OG1"