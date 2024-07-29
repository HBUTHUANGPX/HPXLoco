import pinocchio as pin
from pinocchio.utils import *
import numpy as np

np.set_printoptions(precision=5, suppress=True, linewidth=100000, threshold=100000)

model = pin.buildModelFromUrdf(
    "/home/hpx/HPXLoco/livelybot_GYM_control/assets/urdf/pai_12dof/urdf/pai_12dof.urdf"
)
print("model name: " + model.name)
data = model.createData()

q = np.zeros(model.nq)
print(model.nq)
# 计算前向运动学
pin.forwardKinematics(model, data, q)
pin.framesForwardKinematics(model, data, q)


for name, oMi in zip(model.names, data.oMi):
    print(("{:<24} : {: .6f} {: .6f} {: .6f}".format(name, *oMi.translation.T.flat)))

# end_effector_id = model.getJointId("")
for joint_id in range(model.njoints):
    joint_name = model.names[joint_id]
    print(f"Joint {joint_id}: {joint_name}")

print(pin.computeTotalMass(model))

M = pin.crba(model, data, q)

# 打印惯性矩阵
joint_indices = [[5, 4, 3, 2, 1, 0], [11, 10, 9, 8, 7, 6]]
print("Inertia matrix M(q):")
print(M)
print(M[np.ix_(joint_indices[0], joint_indices[0])])
print(M[np.ix_(joint_indices[1], joint_indices[1])])

baselink_frame_id = model.getFrameId("base_link")
print("baselink_frame_id: ", baselink_frame_id)
r_ankle_roll_frame_id = model.getFrameId("r_ankle_roll_link")
l_ankle_roll_frame_id = model.getFrameId("l_ankle_roll_link")
print("r_ankle_roll_frame_id: ", r_ankle_roll_frame_id)
J_baselink_to_r_ankle_roll_link = pin.computeFrameJacobian(
    model, data, q, r_ankle_roll_frame_id, pin.ReferenceFrame.LOCAL
)[:,6:]
print("Jacobian matrix from baselink to r_ankle_roll:")
print(J_baselink_to_r_ankle_roll_link)
J_baselink_to_l_ankle_roll_link = pin.computeFrameJacobian(
    model, data, q, l_ankle_roll_frame_id, pin.ReferenceFrame.LOCAL
)[:,:6]
print("Jacobian matrix from baselink to l_ankle_roll:")
print(J_baselink_to_l_ankle_roll_link)
J_r_ankle_roll_to_baselink = np.linalg.pinv(J_baselink_to_l_ankle_roll_link)
print("Jacobian matrix from ankle_roll to baselink:")
print(J_r_ankle_roll_to_baselink)

feet_index = {"left":[],"right":[]}
print(feet_index)

# 获取关节的 SE(3) 变换
joint_id = model.getJointId("r_ankle_roll_joint")
joint_placement = data.oMf[joint_id]
print(joint_placement.rotation)
print(joint_placement.translation)
# 打印 4x4 矩阵
print("4x4 transformation matrix:")
R_ab = joint_placement.rotation
t_ab = joint_placement.translation
R_ba = R_ab.T
print(R_ba,R_ab)
t_ba = -R_ba @ t_ab
T_ba = pin.SE3(R_ba, t_ba)
print(joint_placement.homogeneous)
print(joint_placement.inverse().homogeneous)
print(T_ba.homogeneous)