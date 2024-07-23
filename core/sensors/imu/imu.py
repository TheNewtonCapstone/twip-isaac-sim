# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import Dict

from core.sensors.imu.imu_data import IMUData
from core.sensors.sensor_base import SensorBase
from core.utils.usd import find_first_matching_prim
import core.utils.math as math_utils


class IMU(SensorBase):
    """The inertia measurement unit sensor.

    The sensor can be attached to any RigidObject in the scene, which becomes the reference frame. The sensor provides the linear acceleration and angular
    velocity of the object in the reference frame. The sensor also provides the orientation of the object in the world frame.
    """

    def __init__(self, config: Dict):
        """Initializes the IMU sensor.

        Args:
            cfg: The configuration parameters.
        """
        # initialize base class
        super().__init__(config)

        self.offset = {
            "pos": self.config["offset"]["pos"],
            "rot": self.config["offset"]["rot"],
        }

        # Create empty variables for storing output data
        self._data = IMUData()

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"IMU sensor @ '{self.prim_path}': \n"
            f"\tview type         : {self._view.__class__}\n"
            f"\tupdate period (s) : {self.update_period}\n"
            f"\tnumber of sensors : {self._view.count}\n"
        )

    """
    Properties
    """

    @property
    def data(self) -> IMUData:
        # update sensors if needed
        self._update_outdated_buffers()

        # return the data
        return self._data

    @property
    def num_instances(self) -> int:
        return self._view.count

    """
    Operations
    """

    def reset(self, env_ids: torch.Tensor | None = None):
        # reset the timestamps
        super().reset(env_ids)

        # resolve None
        if env_ids is None:
            env_ids = torch.zeros(self._num_envs)

        # reset accumulative data buffers
        self._data.quat_w[env_ids] = 0.0
        self._data.ang_vel_b[env_ids] = 0.0
        self._data.lin_acc_b[env_ids] = 0.0

    def update(self, dt: float, force_recompute: bool = False):
        # save timestamp
        self._dt = dt

        # execute updating
        super().update(dt, force_recompute)

    """
    Implementation.
    """

    def _initialize_impl(self):
        """Initializes the sensor handles and internal buffers.

        This function creates handles and registers the provided data types with the replicator registry to
        be able to access the data from the sensor. It also initializes the internal buffers to store the data.

        Raises:
            RuntimeError: If the number of imu prims in the view does not match the number of environments.
            RuntimeError: If the imu prim is not a RigidBodyPrim
        """

        import omni.physics.tensors.impl.api as physx
        from pxr import UsdPhysics

        # Initialize parent class
        super()._initialize_impl()

        # create simulation view
        self._physics_sim_view = physx.create_simulation_view(self._backend)
        self._physics_sim_view.set_subspace_roots("/")

        # check if the prim at path is a rigid prim
        prim = find_first_matching_prim(self.prim_path)
        if prim is None:
            raise RuntimeError(
                f"Failed to find a prim at path expression: {self.prim_path}"
            )

        # check if it is a RigidBody Prim
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            self._view = self._physics_sim_view.create_rigid_body_view(
                self.prim_path.replace(".*", "*")
            )
        else:
            raise RuntimeError(
                f"Failed to find a RigidBodyAPI for the prim paths: {self.prim_path}"
            )

        # Create internal buffers
        self._initialize_buffers_impl()

    def _update_buffers_impl(self, env_ids: torch.Tensor):
        """Fills the buffers of the sensor data."""

        # check if self._dt is set (this is set in the update function)
        if not hasattr(self, "_dt"):
            raise RuntimeError(
                "The update function must be called before the data buffers are accessed the first time."
            )

        # obtain the poses of the sensors
        pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], dim=-1)
        quat_w = math_utils.convert_quat(quat_w, to="wxyz")

        # store the poses
        # self._data.pos_w[env_ids] = pos_w + math_utils.quat_rotate(quat_w, self._offset_pos)
        self._data.quat_w[env_ids] = math_utils.quat_mul(quat_w, self._offset_quat)

        # obtain the velocities of the sensors
        lin_vel_w, ang_vel_w = self._view.get_velocities()[env_ids].split(
            [3, 3], dim=-1
        )

        # store the velocities
        # note: we clone here because the obtained tensors are read-only
        self._data.ang_vel_b[env_ids] = math_utils.quat_rotate_inverse(
            self._data.quat_w[env_ids], ang_vel_w
        )
        self._data.lin_acc_b[env_ids] = math_utils.quat_rotate_inverse(
            self._data.quat_w[env_ids],
            (lin_vel_w - self._last_lin_vel_w[env_ids])
            / max(self._dt, self.update_period),
        )
        self._last_lin_vel_w[env_ids] = lin_vel_w.clone()

    def _initialize_buffers_impl(self):
        """Create buffers for storing data."""

        # data buffers
        self._data.pos_w = torch.zeros(self._view.count, 3, device=self._device)
        self._data.quat_w = torch.zeros(self._view.count, 4, device=self._device)
        self._data.quat_w[:, 0] = 1.0
        self._data.lin_acc_b = torch.zeros(self._view.count, 3, device=self._device)
        self._data.ang_vel_b = torch.zeros(self._view.count, 3, device=self._device)

        # internal buffers
        self._last_lin_vel_w = torch.zeros(self._view.count, 3, device=self._device)

        # store sensor offset transformation
        self._offset_pos = torch.tensor(
            list(self.offset["pos"]), device=self._device
        ).repeat(self._view.count, 1)
        self._offset_quat = torch.tensor(
            list(self.offset["rot"]), device=self._device
        ).repeat(self._view.count, 1)
