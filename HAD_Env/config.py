import numpy as np
from common.arguments import get_args


# 1. 可视化的相关设置

# 1.1 可视化显示界面设置
LENGTH = 800  # 渲染界面长度
WIDTH = 800  # 渲染界面宽度
HEIGHT = int(800 * 0.1)  #渲染界面高度,为了方便显示缩放10倍距离

# 1.2 颜色
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
PURPLE = (210, 191, 255)
GREEN = (0, 255, 0)
GR = (153, 204, 153)
BL2 = (153, 204, 255)
BL1 = (204, 255, 255)
BL = (0, 0, 255)


# 2. 环境中智能体设置

# 2.1 初始化位置设置
AttackRatio = 0.3
DisturbRatio = 0.2
ScoutRatio = 0.1
ProtectRatio = 0.05

# 2.2 初始化属性设置

# 2.2.1 目标点属性
initial_health = 1.2  # 目标点初始健康值

# 2.2.2 通用智能体属性
AvoidanceDistance = 2  # 避碰常数
vDomain = [20., 250.]  # 最大速度
aMax = 30.  # 最大加速度
BlueAmaxCoef = 1.2
wMax = np.pi  # 最大角速度

# 2.2.3 打击智能体属性
AttackDistance = [500., 500.]  # 打击成功概率距离参数
AttackIntensity = 10  # 打击强度
RedAttackCoef = 0.25  # 红方智能体在打击时计算杀伤比例时己方的衰减系数

# 2.2.4 干扰智能体属性
DisturbAngleMax = np.pi / 3.  # 干扰角度
DisturbDistanceMax = 600.  # 干扰最大距离
DisturbIntensity = 0.1  # 干扰强度
GaussSigma = 1.  # 干扰强度的高斯分布参数
DisturbStopDistanceRatio = 0.8  # 规则智能体下的最佳干扰距离比例

# 2.2.5 侦查智能体属性
ScoutAngleMax = np.pi * 2 / 3.  # 侦查角度
ScoutDistanceMax = 2000.  # 侦查最大距离


# 3. 环境属性

# 3.1 环境基础属性
Boundary = True  # 环境是否有边界
Interval = 1  # 每步间的时间间隔
AeroPoint = [[-2500., 2500.], [-2500., 2500.], [0, 1000]]  # 坐标限制，分别是x,y,z三轴
def get_area_point(area_vector, ratio):
    assert 0 <= ratio <= 1
    return area_vector[0] + (area_vector[1] - area_vector[0]) * ratio
EnvDim = len(AeroPoint)

# 3.2 环境奖励属性
reward_disturb_single = 0.  # 软杀伤1点健康值获得的奖励
reward_scout_single = 0.  # 侦查到单个敌人获得的奖励
reward_attack_single = 0.  # 打击1点健康值获得的奖励

reward_boundary = 0.05  # 靠近边界时每interval获得的惩罚

reward_episode = 10

ScreenLength = 800  # 渲染界面长度
ScreenWidth = ScreenLength * (AeroPoint[1][1] - AeroPoint[1][0]) / (AeroPoint[0][1] - AeroPoint[0][0])  # 渲染界面宽度
ScreenHeight = int(ScreenLength * (AeroPoint[2][1] - AeroPoint[2][0]) / (AeroPoint[0][1] - AeroPoint[0][0]) * 0.5)  #渲染界面高度,为了方便显示缩放10倍距离

# 1.2 颜色（含透明度）
SurfaceColor = (200, 220, 255, 20)   # 背景色：蓝色
BorderColor = (150, 200, 255, 50)          # 边框：不透明黑色

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