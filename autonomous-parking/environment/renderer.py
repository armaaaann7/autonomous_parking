"""Pygame renderer (blueprint Part 27). Works headless: set SDL_VIDEODRIVER=dummy before
importing pygame to render to an off-screen surface (used by tests and video export) with the
same code path as a live window.
"""
from __future__ import annotations

import numpy as np
import pygame

PPM = 40  # pixels per metre
BG = (30, 32, 36)
CAR = (66, 165, 245)
SLOT = (102, 187, 106)
OBSTACLE = (120, 80, 80)
RAY_OK = (90, 200, 120)
RAY_WARN = (230, 200, 80)
RAY_DANGER = (230, 90, 90)
TEXT = (230, 230, 230)
SHIELD_ON = (230, 90, 90)


class Renderer:
    def __init__(self, env, mode="human", width=800, height=800):
        self.env = env
        self.mode = mode
        self.width, self.height = width, height
        pygame.font.init()
        if mode == "human":
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Autonomous Parking")
        else:
            self.screen = pygame.Surface((width, height))
        self.font = pygame.font.SysFont("consolas", 16)
        self.trail = []
        self.shield_active = False

    def _to_px(self, x, y):
        return int(self.width / 2 + x * PPM), int(self.height / 2 - y * PPM)

    def _rect_px(self, aabb):
        x0, y0 = self._to_px(aabb[0], aabb[3])
        x1, y1 = self._to_px(aabb[2], aabb[1])
        return pygame.Rect(x0, y0, x1 - x0, y1 - y0)

    def set_shield_active(self, active: bool):
        self.shield_active = bool(active)

    def draw(self):
        env = self.env
        self.screen.fill(BG)
        scn = env.scenario

        # slot outline
        from .scenarios import slot_rectangle
        pygame.draw.rect(self.screen, SLOT, self._rect_px(slot_rectangle(scn, env.vehicle.length,
                                                                          env.vehicle.width)), width=2)
        # obstacles
        for obs in scn.obstacles:
            pygame.draw.rect(self.screen, OBSTACLE, self._rect_px(obs))

        # trajectory trail
        self.trail.append((env.vehicle.x, env.vehicle.y))
        if len(self.trail) > 400:
            self.trail.pop(0)
        if len(self.trail) > 1:
            pts = [self._to_px(x, y) for x, y in self.trail]
            pygame.draw.lines(self.screen, (90, 90, 110), False, pts, 2)

        # sensor rays, colour-coded by distance
        from .sensors import cast_rays
        rays = cast_rays(env.vehicle.x, env.vehicle.y, env.vehicle.theta, scn.obstacles,
                          n_rays=env.n_rays, max_range=env.sensor_range)
        angles = env.vehicle.theta + np.linspace(0, 2 * np.pi, env.n_rays, endpoint=False)
        for ang, d in zip(angles, rays):
            ex, ey = env.vehicle.x + d * np.cos(ang), env.vehicle.y + d * np.sin(ang)
            color = RAY_OK if d > 3 else (RAY_WARN if d > 1.5 else RAY_DANGER)
            pygame.draw.line(self.screen, color, self._to_px(env.vehicle.x, env.vehicle.y),
                              self._to_px(ex, ey), 1)

        # car body
        corners = [self._to_px(*c) for c in env.vehicle.corners()]
        pygame.draw.polygon(self.screen, CAR, corners)
        nose = env.vehicle.corners()[:2].mean(axis=0)
        pygame.draw.line(self.screen, (255, 255, 255), self._to_px(env.vehicle.x, env.vehicle.y),
                          self._to_px(*nose), 2)

        self._hud()
        if self.mode == "human":
            pygame.display.flip()
            return None
        return pygame.surfarray.array3d(self.screen).swapaxes(0, 1)

    def _hud(self):
        env = self.env
        lines = [
            f"steer {env.vehicle.delta / max(env.vehicle.max_steer,1e-9):+.2f}  "
            f"v {env.vehicle.v:+.2f} m/s",
            f"pos_err {env._dist_to_slot():.2f} m   step {env.steps}/{env.max_steps}",
            f"safety shield: {'INTERVENING' if self.shield_active else 'idle'}",
        ]
        for i, line in enumerate(lines):
            color = SHIELD_ON if (i == 2 and self.shield_active) else TEXT
            surf = self.font.render(line, True, color)
            self.screen.blit(surf, (10, 10 + 18 * i))

    def close(self):
        if self.mode == "human":
            pygame.display.quit()
