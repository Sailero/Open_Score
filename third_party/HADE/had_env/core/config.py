"""HAD 环境的默认参数。

距离单位：米；速度：米/秒；加速度：米/秒²；角度：弧度；时间：秒。
本文件由独立 HAD 环境和 Open-SCORE 共用，网页没有另一份物理配置。
修改后需重新启动已加载环境的 Python / 网页服务，新建环境才会使用新参数；
已经保存的回放不会改变。构造环境时显式传入的参数优先于这里的默认值。
二维/三维在 make_env(..., spatial_dim=2 或 3) 选择，不要通过修改 EnvDim 切换。
"""

import numpy as np


# 1. 可视化的相关设置

# 1.1 旧渲染接口保留的像素尺寸；当前原生渲染用文件末尾的 Screen 系列。
# 网页图像尺寸由 viewer.html 决定，不由这些像素参数控制，也不影响物理距离。
LENGTH = 800  # 旧画布横向尺寸，像素
WIDTH = 800  # 旧画布纵向尺寸，像素
HEIGHT = int(800 * 0.1)  # 旧高度投影区域的像素尺寸，并非场地高度

# 1.2 RGB 颜色；三个数依次为红、绿、蓝通道，范围均为 0–255。
BLUE = (0, 0, 255)  # 蓝色
WHITE = (255, 255, 255)  # 白色
RED = (255, 0, 0)  # 红色
BLACK = (0, 0, 0)  # 黑色
PURPLE = (210, 191, 255)  # 淡紫色
GREEN = (0, 255, 0)  # 绿色
GR = (153, 204, 153)  # 灰绿色，旧绘图配色
BL2 = (153, 204, 255)  # 浅蓝色，旧绘图配色
BL1 = (204, 255, 255)  # 淡青色，旧绘图配色
BL = (0, 0, 255)  # 蓝色，旧绘图别名


# 2. 环境中智能体设置

# 2.1 仅用于 initialization="uniform" 的无人机排列；默认随机出生不使用这些比例。
# 数值是出生位置距己方 x 边界占场地宽度的比例，不是各类智能体的数量比例。
AttackRatio = 0.3  # 攻击机距己方 x 边界 30%；当前场地对应红 x=-1000、蓝 x=1000
DisturbRatio = 0.2  # 干扰机距己方 x 边界 20%；红 x=-1500、蓝 x=1500
ScoutRatio = 0.1  # 侦察机距己方 x 边界 10%；红 x=-2000、蓝 x=2000
ProtectRatio = 0.05  # 旧参数，当前目标点初始化不读取它；请改 DefaultTargetRegion

# 2.2 初始化属性设置

# 2.2.1 目标点属性
initial_health = 2.0  # 每个目标的默认血量；可用 make_env(target_health=...) 或网页覆盖
# survival：血量耗尽即失守，默认两次满伤命中摧毁目标。
# damage：目标血量保持不变，只累加伤害；红方回报 -伤害、蓝方 +伤害。
# 无人机血量当前为 1，在 agents/base.py 中初始化，不由 initial_health 控制。

# 2.2.2 通用智能体属性
AvoidanceDistance = 20.  # 两无人机同一步、同一时刻间距 <=20 米判碰撞；敌我双方都适用
# 这是两机中心之间的距离阈值，不是单机半径；目标点不参与这项无人机碰撞检测。
vDomain = [35., 120.]  # [双方最低速度, 红方最高速度]，米/秒；常规飞行不能主动悬停
aMax = 40.  # 红方加速度控制尺度，米/秒²；标准离散动作已归一化
BlueAmaxCoef = 1.0  # 蓝方加速度尺度 = aMax × 本系数；当前与红方相同
BlueVmaxCoef = 1.0  # 蓝方最高速度 = vDomain[1] × 本系数；最低速度仍为 vDomain[0]
wMax = np.pi  # 每步最大转向角，弧度；π=180°，当前不额外限制转向（并非每秒角速度）

# 2.2.3 打击智能体属性
AttackDistance = {
    2: [200., 400.],  # 二维：200 米内满伤，200–400 米线性衰减，>=400 米无伤害
    3: [300., 600.],  # 三维：300 米内满伤，300–600 米线性衰减，>=600 米无伤害
}
# 每个环境按 spatial_dim 取一组，存入 env.attack_distance，二维/三维实例互不覆盖。
# 自动开火阈值 env.fire_range 默认等于该组第一个数，距离严格小于阈值才开火。
# 红方以蓝方攻击机触发开火，蓝方以目标点触发开火；友军不触发，也不受爆炸伤害。
# 攻击机开火后自身消亡；第二个半径描述爆炸波及范围，不是主动开火阈值。
AttackIntensity = 1.0  # 每个满伤受击实体扣 1 点；伤害区间内的实体分别按距离结算


def attack_distance_for(spatial_dim=2):
    """按维度读取并校验半径，返回 (满伤半径, 归零半径)，不改写全局配置。"""
    if isinstance(spatial_dim, bool) or spatial_dim not in (2, 3):
        raise ValueError("spatial_dim must be 2 or 3")
    inner, outer = map(float, AttackDistance[spatial_dim])
    if not np.isfinite([inner, outer]).all() or not 0 < inner < outer:
        raise ValueError("AttackDistance must satisfy 0 < full-damage radius < zero-damage radius")
    return inner, outer


RedAttackCoef = 0.25  # 已弃用：旧密度开火判据系数，保留以免外部引用断裂

# 2.2.4 干扰智能体属性
# 只有创建了干扰机才使用；当前网页仅有攻击机，不使用这一组。
DisturbAngleMax = np.pi / 3.  # 干扰扇区总夹角 60°；以速度方向为中心，左右各 30°
DisturbDistanceMax = 600.  # 干扰作用的最大距离，米；还需满足扇区条件
DisturbIntensity = 0.1  # 每步软杀伤强度系数，实际伤害还乘高斯距离衰减
GaussSigma = 1.  # 高斯距离衰减的标准差，米；当前为 1，离开数米后软杀伤已很小
DisturbStopDistanceRatio = 0.8  # 旧规则预留参数，当前执行路径未使用，不影响飞行

# 2.2.5 侦查智能体属性
# 只有创建了侦察机才使用；当前网页仅有攻击机，不使用这一组。
ScoutAngleMax = np.pi * 2 / 3.  # 侦察扇区总夹角 120°；以速度方向为中心，左右各 60°
ScoutDistanceMax = 2000.  # 侦察最大距离，米；控制侦察可见性，不造成攻击伤害


# 3. 环境属性

# 3.1 环境基础属性
Boundary = True  # True：位置越界时裁剪到边界，并消除朝外的速度分量；越界本身不会判死
Interval = 1  # 一个物理步代表 1 秒；先按旧速度移动，再用本步加速度更新速度
AeroPoint = [
    [-2500., 2500.],  # x 轴范围，米；负 x 为目标所在的左侧，正 x 为蓝方来袭侧
    [-2500., 2500.],  # y 轴范围，米；平面上下方向
    [0., 2500.],  # z 轴范围，米；三维高度边界，二维固定平面也必须在此范围内
]

# 目标点随机出生区域，不是整张地图范围；每个目标分别采样，生成后保持静止。
# x/y/z 各按对应区间均匀采样；二维只采样 x/y，高度统一用 PlanarAltitude。
# 相同 seed 和完整配置可复现；重新指定同一个 seed，不代表换一套随机布局。
# target_initialization="random"（训练默认）每局重新抽样；"fixed" 使用固定布局。
# fixed 默认布局：x/z 取区域中点，y 在区域内部等间距排列；二维 z 仍取平面高度。
# 显式传入 target_region 会覆盖此区域；传 target_positions 可指定固定坐标。
DefaultTargetRegion = [
    [-2300., -1900.],  # 目标 x：地图左侧一条 400 米宽的区域
    [-1200., 1200.],  # 目标 y：左右侧向分布范围共 2400 米
    [500., 1500.],  # 目标 z：仅三维有效，和无人机的 SpawnAltitude 分开设置
]
SpawnAltitude = [200., 1500.]  # 三维无人机初始高度随机区间，米；不控制目标高度
PlanarAltitude = 100.  # 二维所有无人机和目标统一 z=100 米，垂直速度、加速度均为 0
PlanarSpawnSeparation = 30.  # 二维初始化时拒绝过近布局；初始实体间距需大于此值，米
# 初始化间距至少还要大于 AvoidanceDistance，只防止开局重叠，不防止后续相撞。
RedSpawnAnnulus = [800., 2200.]  # 随机选一个目标，红机在其 XY 平面周围 800–2200 米环带出生
# 环带按面积均匀采样；二维越界位置重新抽样，三维位置裁剪到场地边界。
BlueSpawnX = [0., 2500.]  # 随机出生时蓝机的 x 区间；y 使用整个 AeroPoint[1]
DefaultMaxSteps = 100  # 网页 / run_episode 默认物理步数上限；100 步 × Interval=100 秒
# 原生 make_env 的时限由 max_cycles 控制；传入 max_steps/max_cycles 或网页值优先。
HorizonPolicy = "red_win"  # 分组评估 survival 到时记红胜；还可选 "draw" / "blue_win"
# damage 到时只截断并汇总 ±累计伤害，不加胜负奖励；Parallel 的 max_cycles 也仅作截断。
# 该默认值只在读取它的入口生效，显式 horizon_policy 参数会覆盖它。
OBS_ENTITY_DIM = 11  # 每个实体的观测列数：相对位置3+相对速度3+血量1+存活1+阵营标志3
# 二维也保留三个坐标槽，z 相对位置和垂直速度为 0；不要直接改这个常量来裁剪观测。
def get_area_point(area_vector, ratio):
    """在 [下界, 上界] 内按 0–1 比例插值；例如 ratio=0.5 返回中点。"""
    assert 0 <= ratio <= 1
    return area_vector[0] + (area_vector[1] - area_vector[0]) * ratio
EnvDim = len(AeroPoint)  # 内部坐标槽数，始终为 3；实际运动维度由 spatial_dim 决定

# 3.2 survival 的辅助奖励；damage 直接返回 ±本步新增目标伤害，不读取这些权重。
reward_disturb_single = 0.  # 干扰机个体辅助奖励系数；0 表示关闭该项
reward_scout_single = 0.  # 侦察机每个观测到的实体的辅助奖励系数；0 表示关闭该项
reward_attack_single = 0.  # 旧攻击奖励权重预留；当前攻击机的几何辅助奖励未乘此系数

reward_boundary = 0.05  # 侦察/干扰机贴近 XY 边界时的辅助惩罚系数，不是碰撞伤害

reward_episode = 10  # 原生 survival 终局团队奖励尺度：红胜红+10/蓝-10，蓝胜相反
# 分组适配器自己的 survival 结果使用 ±1；damage 全部使用未经此系数缩放的伤害回报。

# 4. 原生 Pygame 渲染尺寸；只影响显示，不影响物理距离，也不控制网页布局。
ScreenLength = 800  # 原生图横向宽度，像素
ScreenWidth = ScreenLength * (AeroPoint[1][1] - AeroPoint[1][0]) / (AeroPoint[0][1] - AeroPoint[0][0])  # XY 图纵向像素数，保持 x/y 比例
ScreenHeight = int(ScreenLength * (AeroPoint[2][1] - AeroPoint[2][0]) / (AeroPoint[0][1] - AeroPoint[0][0]) * 0.5)  # 高度投影区域像素数，额外压缩为原比例的一半

# RGBA 颜色：末项是透明度，0 完全透明、255 不透明。
SurfaceColor = (200, 220, 255, 20)   # 浅蓝背景，低不透明度
BorderColor = (150, 200, 255, 50)  # 浅蓝边框，半透明

# 红方配色（Red 系列）
RedControlColor = (255, 102, 102, 255)         # 深红色，不透明
RedColor = (255, 153, 153, 255)          # 中红色，不透明
RedControlSurfaceColor = (255, 235, 235, 100)  # 非常浅的红色，半透明

# 蓝方配色（Blue 系列）
BlueControlColor = (102, 153, 255, 255)         # 深蓝色，不透明
BlueColor = (153, 204, 255, 255)          # 中蓝色，不透明
BlueControlSurfaceColor = (235, 245, 255, 100)  # 非常浅的蓝色，半透明

# 阵亡统一颜色（灰色）
DeadControlColor = (192, 192, 192, 180)   # 稍透明
DeadAgentColor = (192, 192, 192, 180)   # 稍透明
