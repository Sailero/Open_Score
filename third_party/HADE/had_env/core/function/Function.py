from scipy.stats import norm
from scipy.spatial.transform import Rotation
from had_env.core.config import *
import numpy as np


# 计算两个矢量之间的距离
def distance(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.sqrt(np.sum((p1 - p2) ** 2))


def distances_from(position, positions):
    """Batch the same subtraction/square/sum/sqrt used by ``distance``.

    Homogeneous world coordinates share one small NumPy reduction. Mixed
    coordinate dtypes or shapes retain the scalar path so batching cannot
    silently promote a previously lower-precision pair or alter broadcasting.
    """
    points = [np.asarray(point) for point in positions]
    if not points:
        return np.empty(0, dtype=np.float64)
    first = points[0]
    observer = np.asarray(position)
    if observer.ndim != 1 or first.ndim != 1 or any(point.dtype != first.dtype or point.shape != first.shape
                              for point in points[1:]):
        return np.asarray([distance(position, point) for point in points])
    offsets = observer - np.asarray(points)
    return np.sqrt(np.sum(offsets ** 2, axis=1))


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
def attack_intensity_ratio(d, attack_distance=None):
    inner, outer = attack_distance_for(2) if attack_distance is None else attack_distance
    if isinstance(d, float):
        if d < inner:
            return 1
        elif d >= outer:
            return 0
        else:
            return (outer - d) / (outer - inner)
    elif isinstance(d, (list, np.ndarray)):
        result = []
        for element in d:
            if element < inner:
                result.append(1)
            elif element >= outer:
                result.append(0)
            else:
                result.append((outer - element) / (outer - inner))
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


def _orthogonal_axis(vector):
    """Deterministic unit axis orthogonal to ``vector`` via Gram-Schmidt."""
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    unit = value / norm
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(unit[0]) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    axis = np.cross(unit, helper)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-9:
        axis = np.cross(unit, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        axis_norm = float(np.linalg.norm(axis))
    return axis / max(axis_norm, 1e-12)


# 给出此刻速度与下一时刻的速度，若下一时刻速度偏向角度超过wMax，则按照先前旋转趋势强行转换为偏转wMax角度的向量，但速度大小不变
# 当前 wMax=π，arccos 值域即为 [0,π]，该限制分支不会进入。
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
            axis = _orthogonal_axis(v1)
        angle = wMax

        # 强行将速度变为wMax的方向并与下一时刻速度大小保持一致
        v2 = rotate_vector(v1, axis, angle) / np.linalg.norm(v1) * np.linalg.norm(v2)
    return v2.tolist()


def closest_segment_distance(p1, q1, p2, q2):
    """Minimum simultaneous separation over one linear-motion step.

    Both trajectories must use the same normalized time in [0, 1]. Allowing
    independent segment parameters falsely collides agents arriving at the
    same location at different times.
    """
    p1, q1, p2, q2 = (np.asarray(item, dtype=np.float64) for item in (p1, q1, p2, q2))
    relative_position = p1 - p2
    relative_displacement = (q1 - p1) - (q2 - p2)
    speed2 = float(np.dot(relative_displacement, relative_displacement))
    time = (0.0 if speed2 <= 1e-12 else
            float(np.clip(-np.dot(relative_position, relative_displacement) / speed2, 0.0, 1.0)))
    return float(np.linalg.norm(relative_position + time * relative_displacement))


def world_diagonal():
    return float(np.sqrt(sum((AeroPoint[dim][1] - AeroPoint[dim][0]) ** 2 for dim in range(EnvDim))))


def velocity_scale():
    return 2.0 * float(vDomain[1]) * float(BlueVmaxCoef)


class WorldKinematics:
    """Lightweight pre-update snapshot used instead of deepcopy(world)."""

    __slots__ = ("positions", "velocities", "health", "is_fire", "is_disturb",
                 "color_code", "type_code", "attack_distance")

    def __init__(self, world, color_code=None, type_code=None, *, attack_distance=None):
        self.attack_distance = tuple(attack_distance if attack_distance is not None else
                                     world[0].attack_distance if world else attack_distance_for(2))
        self.positions = np.asarray([entity.position for entity in world], dtype=np.float64)
        self.velocities = np.asarray([entity.velocity for entity in world], dtype=np.float64)
        self.health = np.asarray([float(entity.Health) for entity in world], dtype=np.float64)
        self.is_fire = np.asarray([bool(getattr(entity, "IsFire", False)) for entity in world], dtype=bool)
        self.is_disturb = np.asarray([bool(getattr(entity, "IsDisturb", False)) for entity in world], dtype=bool)
        if color_code is None:
            mapping = {"Red": 0, "Blue": 1, "Entity": 2}
            color_code = np.asarray([mapping[entity.Color] for entity in world], dtype=np.int8)
        if type_code is None:
            mapping = {"Entity": 0, "Attack": 1, "Disturb": 2, "Scout": 3}
            type_code = np.asarray([mapping[entity.Type] for entity in world], dtype=np.int8)
        self.color_code = color_code
        self.type_code = type_code

    def attack_loss(self, victim_pos, victim_color, victim_type):
        if victim_type == "Entity":
            mask = (self.type_code == 1) & (self.color_code == 1) & (self.health > 0)
        else:
            victim_code = 0 if victim_color == "Red" else 1
            mask = (self.type_code == 1) & (self.color_code != victim_code) & (self.health > 0)
        mask &= self.is_fire
        if not np.any(mask):
            return 0.0
        distances = np.linalg.norm(self.positions[mask] - np.asarray(victim_pos, dtype=np.float64), axis=1)
        return float(AttackIntensity * np.sum(attack_intensity_ratio(distances, self.attack_distance)))

    def disturb_loss(self, victim_pos):
        mask = (self.type_code == 2) & (self.health > 0) & self.is_disturb
        if not np.any(mask):
            return 0.0
        total = 0.0
        victim = np.asarray(victim_pos, dtype=np.float64)
        for position, velocity in zip(self.positions[mask], self.velocities[mask]):
            total += disturb_intensity_ratio(position, victim, velocity)
        return float(DisturbIntensity * total)


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


# 定义状态归一化函数。相对位置除以场地对角线，相对速度除以 2*蓝方vMax。
def normalized_obs(obs_origin):
    obs = [list(row) for row in obs_origin]
    diagonal = world_diagonal()
    speed = velocity_scale()
    for obs_i in range(len(obs)):
        for dim in range(EnvDim):
            obs[obs_i][dim] = obs[obs_i][dim] / diagonal
        obs[obs_i][EnvDim: 2 * EnvDim] = (np.asarray(obs[obs_i][EnvDim: 2 * EnvDim], dtype=np.float64) / speed).tolist()
    return obs


def normalized_obs_n(obs_n):
    return [normalized_obs(obs) for obs in obs_n]
