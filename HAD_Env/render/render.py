import pygame
from pygame.locals import *
from HAD_Env.config import *
import sys

import math
import pygame


def draw_isosceles_triangle(screen, size, center, angle=0., color=(255, 0, 0), top_angle=45):
    """
    绘制一个等腰三角形，顶角朝向指定的角度（angle=0 时表示向右）。

    参数：
    - screen: Pygame 屏幕对象。
    - size: 三角形的高度（从顶点到底边中心）。
    - center: 中心坐标 (x, y)，表示三角形质心。
    - angle: 顶角朝向角度，单位为弧度。0 表示朝右。
    - color: 填充颜色。
    """
    size = size * 2

    # 设置底角为 60 度，即顶角为 60°
    half_base = size * math.tan(math.radians(top_angle / 2))  # 顶角60°

    # 初始方向为“朝右”时的顶点坐标（以 center 为原点）
    vertex_top = (size / 2, 0)  # 顶点（在右侧）
    vertex_left = (-size / 2, -half_base)
    vertex_right = (-size / 2, half_base)

    vertices = [vertex_top, vertex_left, vertex_right]

    # 旋转并平移三角形顶点
    rotated_vertices = []
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    for x, y in vertices:
        x_rot = x * cos_a - y * sin_a + center[0]
        y_rot = x * sin_a + y * cos_a + center[1]
        rotated_vertices.append((x_rot, y_rot))

    pygame.draw.polygon(screen, color, rotated_vertices)


# Base class for the display system
class DisplayPlayer:
    def __init__(self, screen):
        self.screen = screen
        self.interaction_length_range = [int(0.05 * ScreenLength), int(0.95 * ScreenLength)]
        self.interaction_width_range = [int(0.05 * ScreenWidth), int(0.95 * ScreenWidth)]
        self.interaction_height_range = [int(0.05 * ScreenHeight), int(0.95 * ScreenHeight)]
        self.running = True
        self.game_over_font = pygame.font.Font(None, 72)
        self.transparent_layer = pygame.Surface((ScreenLength, ScreenWidth), pygame.SRCALPHA)

    def update(self, agent_info_list, game_result):
        targets_info_n, red_agents_info_n, blue_agents_info_n = agent_info_list
        if not self.running:
            return

        self.handle_events()

        # 使用带透明度的图层绘制各类信息
        transparent_layer = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)

        # 关键中枢点绘制
        self.draw_targets(transparent_layer, targets_info_n)

        # 智能体层
        self.draw_agents(transparent_layer, red_agents_info_n, color="Red")
        self.draw_agents(transparent_layer, blue_agents_info_n, color="Blue")


        # 合成透明层到底层
        self.screen.blit(transparent_layer, (0, 0))

        # 绘制边框
        self.draw_borders()

        self.display_result(game_result)
        pygame.display.flip()
        self.wait_for_result(game_result)

    def handle_events(self):
        """Handle user input events like quitting."""
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.running = False
            elif event.type == QUIT:
                self.running = False

        if not self.running:
            self.close_window()

    def draw_borders(self):
        """Draw the borders of the screen."""
        pygame.draw.line(self.screen, BorderColor,
                         (self.interaction_length_range[0], self.interaction_width_range[0] + self.interaction_height_range[0]),
                         (self.interaction_length_range[1], self.interaction_width_range[0] + self.interaction_height_range[0]),
                         2)
        pygame.draw.line(self.screen, BorderColor,
                         (self.interaction_length_range[0], self.interaction_width_range[0] + self.interaction_height_range[0]),
                         (self.interaction_length_range[0], self.interaction_width_range[1] + self.interaction_height_range[1]),
                         2)
        pygame.draw.line(self.screen, BorderColor,
                         (self.interaction_length_range[0], self.interaction_width_range[1] + self.interaction_height_range[1]),
                         (self.interaction_length_range[1], self.interaction_width_range[1] + self.interaction_height_range[1]),
                         2)
        pygame.draw.line(self.screen, BorderColor,
                         (self.interaction_length_range[1], self.interaction_width_range[1] + self.interaction_height_range[1]),
                         (self.interaction_length_range[1], self.interaction_width_range[0] + self.interaction_height_range[0]),
                         2)
        pygame.draw.line(self.screen, BorderColor,
                         (self.interaction_length_range[0], self.interaction_width_range[1] + self.interaction_height_range[0]),
                         (self.interaction_length_range[1], self.interaction_width_range[1] + self.interaction_height_range[0]),
                         2)

    def draw_targets(self, surface, targets_info_n, size=12):
        target_image = pygame.image.load('./HAD_Env/resources/target.png')
        dead_target_image = pygame.image.load('./HAD_Env/resources/target_dead.png')
        for target_info in targets_info_n:
            agent_px, agent_py, agent_pz = target_info["position"]
            draw_target = target_image if target_info["alive"] else dead_target_image

            center = [get_area_point(self.interaction_length_range, agent_px),
                      get_area_point(self.interaction_width_range, 1 - agent_py) + get_area_point(
                          self.interaction_height_range, 0)]
            # resized_image = pygame.transform.scale(image, (new_width, new_height))
            surface.blit(draw_target, center)
            center_z = [get_area_point(self.interaction_length_range, agent_px),
                        self.interaction_width_range[1] + get_area_point(
                            self.interaction_height_range, agent_pz)]
            pygame.draw.circle(surface, RedColor, center_z, int(size/1.5))

    def draw_agents(self, surface, agents_info_n, color, size=12):
        draw_color = RedColor if color == "Red" else BlueColor
        for agent_info in agents_info_n:
            agent_px, agent_py, agent_pz = agent_info["position"]
            agent_vx, agent_vy, agent_vz = agent_info["velocity"]
            draw_agent_color = draw_color if agent_info["alive"] else DeadAgentColor

            center = [get_area_point(self.interaction_length_range, agent_px),
                      get_area_point(self.interaction_width_range, 1 - agent_py) + get_area_point(self.interaction_height_range, 0)]
            angle = - math.atan2(agent_vy, agent_vx)

            center_z = [get_area_point(self.interaction_length_range, agent_px),
                        self.interaction_width_range[1] + get_area_point(self.interaction_height_range, agent_pz)]
            angle_z = np.pi / 2 if agent_vz < 0 else - np.pi / 2

            if agent_info["type"] == "Attack":
                pygame.draw.circle(surface, draw_agent_color, center, int(size))
                pygame.draw.circle(surface, draw_agent_color, center_z, int(size / 2))
            elif agent_info["type"] == "Disturb":
                draw_isosceles_triangle(surface, size=size, center=center, angle=angle, color=draw_agent_color,
                                        top_angle=45)
                draw_isosceles_triangle(surface, size=int(size / 2), center=center_z, angle=angle_z,
                                        color=draw_agent_color, top_angle=45)
            else:
                draw_isosceles_triangle(surface, size=size, center=center, angle=angle, color=draw_agent_color,
                                        top_angle=36)
                draw_isosceles_triangle(surface, size=int(size / 2), center=center_z, angle=angle_z,
                                        color=draw_agent_color, top_angle=36)

    def display_result(self, game_result):
        if game_result == 1:
            text = "Red Wins"
            color = (255, 0, 0)
        elif game_result == -1:
            text = "Blue Wins"
            color = (0, 0, 255)
        else:
            return
        text_surface = self.game_over_font.render(text, True, color)
        text_rect = text_surface.get_rect(center=(self.screen.get_width() / 2, self.screen.get_height() / 2))
        self.screen.blit(text_surface, text_rect)

    def wait_for_result(self, game_result):
        """Wait for a short duration after displaying the result."""
        start_time = pygame.time.get_ticks()
        if np.abs(game_result):
            while pygame.time.get_ticks() - start_time < 2000:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

    @staticmethod
    def close_window():
        """Close the pygame window."""
        pygame.display.quit()
        pygame.quit()

