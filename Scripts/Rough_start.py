# -*- coding: utf-8 -*-
"""
Created on Tue May  9 01:17:36 2023

@author: agerlt
"""

import time
import numpy as np
from orix.quaternion import Orientation, Symmetry, Quaternion, Rotation
from orix.vector import Vector3d
from typing import List, Optional, Tuple, Union
from itertools import product
import matplotlib.pyplot as plt


def eta_theta(
        incident_beam: Union[List, Tuple, np.ndarray],
        diffracted_beam: Union[List, Tuple, np.ndarray],
        azimuthal_zero: Optional[str] = 'east'
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the eta and theta angles relative to the original beam
    direction and the direction of the diffracted beam. Equivalent to
    equations 1-3 of Joel's diffraction guide.


    Parameters
    ----------
    incident_beam : Union[List, Tuple,np.ndarray]
        a single 3D cartesian vector describing the direction of an incoming
        beam
    diffracted_beam : Union[List, Tuple,np.ndarray]
        Either one or multiple 3D cartesian vectors describing the direction
        in which the incident_beam is being diffracted.
    azimuthal_zero:
        the direction of the zero azimuthal angle TODO: do better
    Returns
    -------
    eta: numpy array of length n
        the aximuthal angle
    theta: numpy array of length n
        the aximuthal
    None.

    """
    # TODO: better description
    # cast i and d to numpy arrays and assert their shape
    i = np.array(incident_beam)
    d = np.array(diffracted_beam)
    assert i.shape(-1) == 3, "incident beam must be an nx3 array of vectors"
    assert d.shape(-1) == 3, "diffracted beam nust be an nx3 array of vectors"
    # normalize the vectors
    i_n = i/np.sum(i*i, axis=-1) ** 0.5
    d_n = d/np.sum(d*d, axis=-1) ** 0.5
    # convert azimuthal direction to a vector
    # TODO: this would be faster as a switch statement...
    azimuthal_dict = {'east': [1, 0, 0],
                      'west': [-1, 0, 0],
                      'north': [0, 1, 0],
                      'south': [0, -1, 0]}
    e = azimuthal_dict[str.lower(azimuthal_zero)]

    # Calc theta (ie, the angle between the two vectors)
    theta = np.arccos(np.dot(d_n, i_n))/2

    # to find the azimuthal angle, we want to find the projections of both
    # the azimuthal zero (e) and the diffraction beam (d_n) onto a plane
    # perpendicular to the beam. The pendactic way to do this is to find the
    # matrix that projects a beam onto the perpendicular plane:
    proj_perp = np.eye(3)-np.outer(i_n, i_n)
    # use it to project the incoming beam:
    d_n_perp = np.dot(proj_perp, d_n.T).T
    # then dot it with the azimuthal zero to find the angle between them
    eta = np.arccos(np.dot(e, d_n_perp.T))
    return(eta, theta)


def eta_theta_fast(
        diffracted_beam: Union[List, Tuple, np.ndarray],
        azimuthal_zero: Optional[str] = 'east'
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    same as eta_theta, but assumes e = [1,0,0] and b = [0,0,1], which makes
    math faster

    Parameters
    ----------
    diffracted_beam : Union[List, Tuple,np.ndarray]
        Either one or multiple 3D cartesian vectors describing the direction
        in which the incident_beam is being diffracted.

    Returns
    -------
    eta: numpy array of length n
        the aximuthal angle
    theta: numpy array of length n
        the aximuthal
    None.

    """
    # TODO: better description
    d = np.array(diffracted_beam)
    assert d.shape(-1) == 3, "diffracted beam nuzst be an nx3 array of vectors"
    # normalize the vectors
    d_n = d/np.sum(d*d, axis=-1) ** 0.5

    # Calc theta (ie, the angle between the two vectors)
    theta = np.arccos(d_n[2])/2
    # Calc azimuthal angle (ie, angle between the projection of the incident
    # beam onto a perpendicular plate, and the east direction on that plane.)
    eta = np.arccos(d_n[0])/2
    return(eta, theta)


def beam_to_lab(b, e):
    """
    Convert beam orientation to lab orientation
    Eqn 5
    """
    R_bl = np.stack([e, np.cross(-b, e), -b])
    return(R_bl)


def detector_to_lab(xyz_det, det_rot, det_trans):
    """translate xyz locations relative to the detector to xyz locations
    relative to the lab reference frame
    Eqn 6
    """
    xyz_lab = np.dot(det_rot, xyz_det) + det_trans
    return(xyz_lab)


def sample_to_lab(xyz_sample, sample_rot, sample_trans):
    """translate xyz locations relative to the sample to xyz locations
    relative to the lab reference frame
    Eqn 7
    """
    xyz_lab = np.dot(sample_rot, xyz_sample) + sample_trans
    return(xyz_lab)


def xtal_to_sample(xyz_xtal, crystal_rot, crystal_loc):
    """translate xyz locations relative to the sample to xyz locations
    relative to the lab reference frame
    Eqn 7
    """
    xyz_sample = np.dot(crystal_rot, xyz_xtal) + crystal_loc
    return(xyz_sample)


def xtal_to_lab(xyz_xtal, crystal_rot, crystal_loc, sample_rot, sample_loc):
    """translate xyz locations relative to the sample to xyz locations
    relative to the lab reference frame
    Eqn 9
    """
    R = np.dot(sample_rot, crystal_rot)
    t = np.dot(sample_rot, crystal_loc) + sample_loc
    xyz_lab = np.dot(R, xyz_xtal) + t
    return(xyz_lab)


# sample rotation and location.
# NOTE: omegas are wrapped up in the sample rotations. so, here you have the
# initial sample rotation, but you also have the axis of rotation and angles
sample_loc = np.array([10, 9, 8])
r_45_100 = Rotation.from_axes_angles([1, 0, 0], np.pi/4)
r_45_010 = Rotation.from_axes_angles([0, 1, 0], np.pi/4)
sample_init_rot = (r_45_100*r_45_010)
# Joel's method is solved for chi being a rotation around only x, which is
# equivalent to saying the axis of rotation is slightly misaligned in only
# the yz plane. this is a bit of an oversimplification, but it makes the
# math WAY easier. also doesn't account for procession during rotation, which
# is probably rarely happening, if ever.
chi = 0.6
chi_rot = Rotation.from_axes_angles([1, 0, 0], chi)
# and finally make all the omega rotations
omega = np.linspace(0, np.pi*2, 144)
omega_rots = Rotation.from_axes_angles([0, 1, 0], omega)
sample_rots = (chi_rot*omega_rots*sample_init_rot).to_matrix()

# xtal rotations and locations (made three orthogonal vector at each corner,
# all rotated by a random rotation)
xtal_loc = np.array([x for x in product([-4, 4], repeat=3)])
r = Rotation.random(8)
# make angles less extreme
xtal_rot = Rotation.from_axes_angles(r.axis, r.angle/10).to_matrix()
xtal_vecs = np.array([[0, 1, 0],
                     [0, 0, 0],
                     [1, 0, 0],
                     [0, 0, 0],
                     [0, 0, 1]])

# detector location and rotation
det_loc = np.array([2, 0, 15])
det_rot = Rotation.from_axes_angles([1, 1, 0], 0.1).to_matrix()

# assume beam and lab align perfectly
beam_loc = np.array([0, 0, 0])
beam_rot = np.eye(3)

# 
plt.close('all')
# check if sample rot/tran works
ax1 = plt.figure().add_subplot(projection='3d')

colors = ([1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1])
for i in range(len(xtal_loc)):
    loc = xtal_loc[i]
    r = xtal_rot[i]
    color = colors[i % 4]
    v_out = []
    for v in xtal_vecs:
        v_out.append(xtal_to_sample(v, r, loc))
    v_out = np.stack(v_out)
    ax1.plot(v_out[:, 0], v_out[:, 1], v_out[:, 2], c=color)

# 

plt.close('all')
ax2 = plt.figure().add_subplot(projection='3d')

colors = ([1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
          [0, 1, 1],
          [1, 1, 0],
          [1, 0, 1],
          [0.5, 0, 1],
          [0, 0.5, 0.5])
for ii, sr in enumerate(sample_rots):
    fract = (ii/len(sample_rots))**0.5
    for i in range(len(xtal_loc)):
        loc = xtal_loc[i]
        r = xtal_rot[i]
        color = colors[i]
        v_out = []
        for v in xtal_vecs:
            a = xtal_to_lab(v, r, loc, sr, sample_loc)
            b = sample_to_lab(xtal_to_sample(v, r, loc), sr, sample_loc)
            assert np.sum((a-b)**2) < 10e-16
            v_out.append(a)
        v_out = np.stack(v_out)
        if ii % 3 == 0:
            col = [c*((fract/2)+0.5) for c in color]
            ax2.plot(v_out[:, 0], v_out[:, 1], v_out[:, 2], c=col)

ax2.plot([0, 10], [0, 0], [0, 0], c='r')
ax2.plot([0, 0], [0, 10], [0, 0], c='g')
ax2.plot([0, 0], [0, 0], [0, 10], c='b')

# %%
# section 2: Diffraction


class Material():
    def __init__(self, a_mag, b_mag, c_mag, alpha, beta, gamma):
        self._a_mag = float(a_mag)
        self._b_mag = float(b_mag)
        self._c_mag = float(c_mag)
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._gamma = float(gamma)
        self._atol = 1e-20

    @property
    def abc(self):
        """
        Converts the 6 values used into define a crystal lattice into
        three vectors defining the unit cell coordinates in crystal
        coordinates.

        NOTE: this follows the convention that X_crystal is parallel to a,
        and Z_crystal is parallel to np.cross(a, b). THis is not the only
        valid interpolation, but it is the one used for all calculations
        in this code.
        """
        if hasattr(self, '_abc') is False:
            ca = np.cos(self._alpha)
            cb = np.cos(self._beta)
            cg = np.cos(self._gamma)
            sa = np.sin(self._alpha)
            sb = np.sin(self._beta)
            sg = np.sin(self._gamma)
            a = self._a_mag
            b = self._b_mag
            c = self._c_mag
            self._abc = np.array([
                [a, 0, 0],
                [b * cg, b * sg, 0],
                [c * cb, c * ca, c * sb * sa]])
            self.abc[(self.abc**2) < self._atol] = 0
        return(self._abc)

    @property
    def vol(self):
        """Unit cell volume. equivalent to a*(b x c)"""
        if hasattr(self, '_volume') is False:
            abc = self.abc
            self._volume = np.dot(abc[0], np.cross(abc[1], abc[2]))
        return(self._volume)

    @property
    def abc_star(self):
        """Reciprical lattice vectors, calculated in crystal coordinates."""
        if hasattr(self, '_abc_star') is False:
            abc = self.abc
            self._abc_star = np.stack([
                np.cross(abc[1], abc[2]),
                np.cross(abc[2], abc[0]),
                np.cross(abc[0], abc[1])
                ])/self.vol
            self._abc_star[(self._abc_star**2) < self._atol] = 0
        return(self._abc_star)

    @property
    def A(self):
        return(self.abc)

    @property
    def B(self):
        return(self.abc_star)

    @property
    def G(self):
        return np.dot(self.abc, self.abc)

    def r2c(self, v_rec):
        """
        transforms a lattice vector in reciprical space to crystal
        coordinates

        Parameters
        ----------
        v_rec : (...,3) dimensional numpy array
            n-dimensional numpy array of 3D vectors in reciprocal space.
            can be any shape, but last dimension must have length 3
        Returns
        -------
        v_crystal: (...,3) dimensional numpy array

        """
        v_rec = np.array(v_rec)
        assert v_rec.shape[-1] == 3, "must be interpretable as an n-by-3 array"
        # take care of trivial cases
        if len(v_rec.shape) <= 1:
            return np.dot(self.B, v_rec)
        if len(v_rec.shape) == 2:
            return(np.dot(self.B, v_rec.T)).T
        # convert to 2d in necessary
        n_vecs = np.prod(v_rec.shape[:-1])
        v_rec_2d = v_rec.reshape(n_vecs, 3)
        v_crystal_2d = np.dot(self.B, v_rec_2d.T).T
        return(v_crystal_2d.reshape(v_rec.shape))

    def r2l(self, v_rec, R_c2s, R_s2l, t_c2s, t_s2l):
        """
        transforms a lattice vector in reciprical space to lab coordinates

        Parameters
        ----------
        v_rec : (...,3) dimensional numpy array
            n-dimensional numpy array of 3D vectors in reciprocal space.
            can be any shape, but last dimension must have length 3
        Returns
        -------
        v_crystal: (...,3) dimensional numpy array

        """
        # initial sanity checks for vectors
        v_rec = np.array(v_rec)
        assert v_rec.shape[-1] == 3, "must be interpretable as an n-by-3 array"
        # initial sanity checks for translation vectors
        t_c2s = np.array(t_c2s, dtype=np.float64).reshape(3)
        t_s2l = np.array(t_s2l, dtype=np.float64).reshape(3)
        # initial sanity checks for rotation matrices
        R_c2s = np.array(R_c2s, dtype=np.float64)
        assert R_c2s.shape == (3, 3), "R_c2s must be 3by3 rotation matrix"
        assert (np.linalg.det(R_c2s)-1)**2 < self._atol, ...
        "R_c2s must be orthogonal"
        # R_s2l checks has to  be a bit more generic, since they can change
        # with, omega and thus we want to be able to request several at once
        assert R_s2l.shape[-2:] == (3, 3), ...
        "R_s2l must be interpretable as 3by3 rotation matrices"
        assert np.mean((np.linalg.det(R_s2l)-1)**2) < self._atol, ...
        "all R_s2l matrices must be orthogonal"

        # pass the trivial case
        if len(R_s2l.shape) == 2:
            v_crystal = self._r2l_single(v_rec, R_c2s, R_s2l, t_c2s, t_s2l)
            return v_crystal
        # otherwise, convert R_s2l to an nx3x3 numpy array
        n_R = np.prod(R_s2l.shape[:-2])
        R_3D = R_s2l.reshape(n_R, 3, 3)
        out = np.stack(
            [self._r2l_single(v_rec, R_c2s, R, t_c2s, t_s2l) for R in R_3D]
            )
        # reshape and return
        v_crystal = out.reshape(R_s2l.shape[:-2] + v_rec.shape)
        return v_crystal

    def _r2l_single(self, v_rec, R_c2s, R_s2l, t_c2s, t_s2l):
        # convert v_rec to 2d array
        if len(v_rec.shape) <= 1:
            v_rec_2d = v_rec.reshape(1, 3)
        if len(v_rec.shape) > 2:
            n_vecs = np.prod(v_rec.shape[:-1])
            v_rec_2d = v_rec.reshape(n_vecs, 3)
        else:
            v_rec_2d = v_rec
        # convert to crystal space
        v_cryst_2d = self.r2c(v_rec_2d)
        # convert from crystal to lab (which is aligned with beam)
        R_c2l = np.dot(R_s2l, R_c2s)
        t_c2l = t_s2l + np.dot(R_s2l, t_c2s)
        v_lab_2d = np.dot(R_c2l, v_cryst_2d.T).T + t_c2l
        # reshape into original form if necessary
        if len(v_rec.shape) <= 1:
            return v_lab_2d.reshape(3)
        return v_lab_2d.reshape(v_rec.shape)

    def c2r(self, v_cryst):
        """
        transforms crystal coordinates into reciprocal lattice vector
        coordinates

        Parameters
        ----------
        v_cryst : (...,3) dimensional numpy array
            n-dimensional numpy array of 3D vectors in crystal lattice
            coordinates.
            can be any shape, but last dimension must have length 3
        Returns
        -------
        v_reciprocal: (...,3) dimensional numpy array

        """
        v_cryst = np.array(v_cryst)
        assert v_cryst.shape[-1] == 3, "invalid array shape."
        # take care of trivial cases
        if len(v_cryst.shape) <= 1:
            return np.dot(self.A, v_cryst)
        if len(v_cryst.shape) == 2:
            return(np.dot(self.B, v_cryst.T)).T
        # convert to 2d in necessary
        n_vecs = np.prod(v_cryst.shape[:-1])
        v_cryst_2d = v_cryst.reshape(n_vecs, 3)
        v_crystal_2d = np.dot(self.A, v_cryst_2d.T).T
        return(v_crystal_2d.reshape(v_cryst.shape))

    def l2r(self, v_lab):
        raise NotImplementedError("""
        Austin said there was no way you would ever need to go lab to
        reciprocal, and he didn't write the code to do so because it looked
        hard. If you are seeing this error, now you know who to blame.""")
        return()


######################
# sample information #
######################
# location
sample_loc = np.array([10, 9, 8])
# tilt
rotation_axis_orientation = Rotation.from_axes_angles([0, 0, 1], 0.3)
# omega spacing
omega = np.linspace(0, np.pi*2, 144)
# create sample rotation matrices
omega_rots = Rotation.from_axes_angles([0, 1, 0], omega)
sample_rots = (chi_rot*omega_rots).to_matrix()

# ###############
# xtal information
# ###############
# define a random material
m = Material(0.5, 0.8, 1.8, np.pi/2, np.pi/5, np.pi/3)
# v = np.random.rand(10000, 3)
# r = Rotation.random(1000).to_matrix()
# t = np.array(sample_loc)

# 
# pick some random crystal locations
# xtal_locs = np.array([x for x in product([-2, 2], repeat=3)])
xtal_locs = np.random.rand(10, 3) * 20
xtal_rots = Rotation.random(10).to_matrix()
# make a unit square at each crystal
xtal_box = np.array([[-1, -1, -1],
                     [-1, -1, 1],
                     [-1, 1, 1],
                     [-1, 1, -1],
                     [-1, -1, -1],
                     [1, -1, -1],
                     [1, -1, 1],
                     [-1, -1, 1],
                     [1, -1, 1],
                     [-1, -1, 1],
                     [1, -1, 1],
                     [1, 1, 1],
                     [-1, 1, 1],
                     [1, 1, 1],
                     [1, 1, -1],
                     [-1, 1, -1],
                     [1, 1, -1],
                     [1, -1, -1]])
xtal_vecs = [np.dot(m.B, (xtal_box + a).T).T for a in xtal_locs]

plt.close('all')
ax2 = plt.figure().add_subplot(projection='3d')
ax2.plot([0, 10], [0, 0], [0, 0], c='r')
ax2.plot([0, 0], [0, 10], [0, 0], c='g')
ax2.plot([0, 0], [0, 0], [0, 10], c='b')


# detector location and rotation
det_loc = np.array([2, 0, 15])
det_rot = Rotation.from_axes_angles([1, 1, 0], 0.1).to_matrix()

# assume beam and lab align perfectly
beam_loc = np.array([0, 0, 0])
beam_rot = np.eye(3)


# plot your rotating crystal
colors = ([1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
          [0, 1, 1],
          [1, 1, 0],
          [1, 0, 1],
          [0.5, 0, 1],
          [0, 0.5, 0.5],
          [0.5, 0, 0.5],
          [0.5, 0.5, 0])
for ii, sr in enumerate(sample_rots):
    fract = (ii/len(sample_rots))**0.5
    for i in range(len(xtal_locs)):
        loc = xtal_locs[i]
        color = colors[i]
        v_out = []
        for v in xtal_box:
            a = xtal_to_lab(v, xtal_rots[i], loc, sr, sample_loc)
            v_out.append(a)
        v_out = np.stack(v_out)
        if ii % 3 == 0:
            col = [c*((fract/2)+0.5) for c in color]
            ax2.plot(v_out[:, 0], v_out[:, 1], v_out[:, 2], c=col)

ax2.plot([0, 10], [0, 0], [0, 0], c='r')
ax2.plot([0, 0], [0, 10], [0, 0], c='g')
ax2.plot([0, 0], [0, 0], [0, 10], c='b')

# 


class Crystal():
    def __init__(self):
        self.a = 1

# add in deformed shape
epsilon = np.random.rand(3,3)*1e-6
U = epsilong = np.eye(3)

recip_2_crystal = np.dot(m.B, np.array([1, 0, 0]))
