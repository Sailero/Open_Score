from scipy.stats import norm
from scipy.spatial.transform import Rotation
from HAD_Env.config import *
import numpy as np


# 计算两个矢量之间的距离
def distance(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.sqrt(np.sum((p1 - p2) ** 2))


# 返回两个向量之间的夹角
def vectors_to_angle(vector1, vector2):
    # 检查向量是否足够接近原点，即它们的长度是否接近零
    if distance(vector1, np.zeros(vector1.shape)) < 1e-3 or distance(vector2, np.zeros(vector2.shape)) < 1e-3:
        return 0.0

    # 计算两个向量的长度
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    # 计算点积
    dot_product = np.dot(vector1, vector2)

    # 规范化点积的结果，确保输入到 arccos 是有效的,不会因为浮点数精度问题导致错误
    cos_theta = dot_product / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1, 1)

    # 计算两个向量之间的角度（以弧度为单位）
    theta = np.arccos(cos_theta)
    return np.abs(theta)


# 计算打击强度的比例
def attack_intensity_ratio(d):  # 可以后续写到AttackAgent类当中
    if isinstance(d, float):
        if d < AttackDistance[0]:
            return 1
        elif d >= AttackDistance[1]:
            return 0
        else:
            return (AttackDistance[1] - d) / (AttackDistance[1] - AttackDistance[0])
    elif isinstance(d, (list, np.ndarray)):
        result = []
        for element in d:
            if element < AttackDistance[0]:
                result.append(1)
            elif element > AttackDistance[1]:
                result.append(0)
            else:
                result.append((AttackDistance[1] - element) / (AttackDistance[1] - AttackDistance[0]))
        return np.array(result)
    else:
        raise ValueError("Input should be a float, list or numpy array.")


# 判断是否在扇形范围内。输入探测智能体与目标智能体的位置，探测智能体的速度，和扇形参数。
# 在扇形范围内返回True，否则为False。
def within_sector_area(position3d1, position3d2, velocity3d, AngleMax, DistMax):
    # position3d1为探测机的坐标；position3d2为被探测机的坐标
    position3d1 = np.array(position3d1)
    position3d2 = np.array(position3d2)
    velocity3d = np.array(velocity3d)
    position3d = position3d2 - position3d1

    # 将角度theta和距离dist与扇形参数做对比，返回bool值
    theta = vectors_to_angle(position3d, velocity3d)
    dist = distance(position3d1, position3d2)
    return (theta < AngleMax / 2) * (dist < DistMax)


# 计算软杀伤比率
def disturb_intensity_ratio(position3d1, position3d2, velocity3d):
    # 输入干扰机与被干扰机的相对位置与干扰机的速度（均为矢量！），基于高斯分布得到干扰强度
    dist = distance(position3d1, position3d2)
    IsDisturbed = within_sector_area(position3d1, position3d2, velocity3d, DisturbAngleMax,
                                     DisturbDistanceMax) * (dist > 1e-3)  # 保证自身不会干扰到自身
    return IsDisturbed * norm.pdf(dist, 0, GaussSigma)


# 计算按照右手螺旋法则旋转angle度后的向量
def rotate_vector(p, axis, angle):
    # p为待旋转向量， axis为旋转轴， angle为旋转度数
    axis = axis / np.linalg.norm(axis)  # 将旋转轴单位化
    r = Rotation.from_rotvec(angle * axis)  # 创建旋转对象
    pnew = r.apply(p)  # 进行向量旋转
    return pnew


# 给出此刻速度与下一时刻的速度，若下一时刻速度偏向角度超过wMax，则按照先前旋转趋势强行转换为偏转wMax角度的向量，但速度大小不变
def rotate_restrict_velocity(v1, v2):
    # v1为此时速度，v2为下一时刻预测速度
    # 计算速度夹角
    if np.linalg.norm(v1) * np.linalg.norm(v2) <= 1e-3:
        angle = 0
    else:
        angle = np.abs(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)))

    # 判断速度夹角是否超过最大偏转角度，如果超过则限制速度夹角为最大偏转角度
    if angle > wMax:
        axis = np.cross(v1, v2).astype(float)
        if np.linalg.norm(axis) < 1e-3:
            axis += np.random.random(3)
        angle = wMax

        # 强行将速度变为wMax的方向并与下一时刻速度大小保持一致
        v2 = rotate_vector(v1, axis, angle) / np.linalg.norm(v1) * np.linalg.norm(v2)
    return v2.tolist()


# 定义智能体距离边界的距离
def boundary_loss(position, boundary_range=100):
    if Boundary:
        min_distance = min(min(abs(position[0] - AeroPoint[0][0]), abs(position[0] - AeroPoint[0][1])),
                           min(abs(position[1] - AeroPoint[1][0]), abs(position[1] - AeroPoint[1][1])))
        if min_distance < boundary_range:
            return reward_boundary * np.exp(-min_distance)
        else:
            return 0
    else:
        return 0


# 定义状态归一化函数
def normalized_obs(obs_origin):
    obs = obs_origin.copy()
    for obs_i in range(len(obs)):
        for dim in range(EnvDim):
            obs[obs_i][dim] = (obs[obs_i][dim] - AeroPoint[dim][0]) / (AeroPoint[dim][1] - AeroPoint[dim][0])
        obs[obs_i][EnvDim: 2 * EnvDim] = (np.array(obs[obs_i][EnvDim: 2 * EnvDim]) - vDomain[0]) / (vDomain[1] - vDomain[0])
    return obs.copy()


def normalized_obs_n(obs_n):
    return [normalized_obs(obs) for obs in obs_n]

