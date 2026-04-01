import math
import random

from config import (
    GW,
    GH,
    BORDER_THICK,
    DIRT,
    ROCK,
    BORDER,
    APHID_MAX_MILK_STOCK,
    CD_MILK,
    APHID_DROP_MIN_DEPTH,
    SOLID_TILES,
    AIR,
    TUNNEL,
)


class Queen:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)


class Spider:
    def __init__(self, x, y):
        self.x, self.y = float(x), float(y)
        self.vx = random.choice([-1.6, 1.6])
        self.vy = 0.0
        self.alive = True
        self.kills = 0

    def update(self, world, ants, phero, stats):
        if not self.alive:
            return
        if ants:
            target = min(ants, key=lambda a: (a.x - self.x) ** 2 + (a.y - self.y) ** 2)
            dx = target.x - self.x
            dy = target.y - self.y
            dist = math.hypot(dx, dy) + 0.001
            self.vx += (dx / dist) * 0.08
            self.vy += (dy / dist) * 0.08
        spd = math.hypot(self.vx, self.vy)
        if spd > 1.2:
            self.vx /= spd / 1.2
            self.vy /= spd / 1.2
        self.vy += 0.15
        nx, ny = self.x + self.vx, self.y + self.vy
        nix, niy = int(nx), int(ny)
        if 0 <= nix < GW and 0 <= niy < GH and world[niy, nix] in (DIRT, ROCK, BORDER):
            self.vy = -0.8
            self.vx *= 0.8
        else:
            self.x, self.y = nx, ny
            self.x = max(BORDER_THICK, min(GW - BORDER_THICK - 1, self.x))
            self.y = max(BORDER_THICK, min(GH - BORDER_THICK - 1, self.y))
        for ant in ants:
            if ant.alive and math.hypot(ant.x - self.x, ant.y - self.y) < 3:
                ant.death_cause = 'spider'
                ant.alive = False
                phero.deposit(ant.x, ant.y, 'danger', 1.0)
                self.kills += 1
                stats['spider_kills'] += 1
                stats['deaths_spider'] += 1


class Aphid:
    def __init__(self, x, y):
        self.x, self.y = float(x), float(y)
        self.alive = True
        self.milk_cooldown = 0
        self.milk_stock = APHID_MAX_MILK_STOCK

    def can_milk(self):
        return self.milk_cooldown <= 0 and self.milk_stock >= 1.0

    def get_milked(self):
        self.milk_cooldown = CD_MILK
        self.milk_stock = max(0.0, self.milk_stock - 1.0)

    def update(self, world, surface_y):
        if self.milk_cooldown > 0:
            self.milk_cooldown -= 1
        ix, iy = int(self.x), int(self.y)
        below_y = iy + 1
        if below_y < GH and world[below_y, ix] in (AIR, TUNNEL):
            self.y = min(GH - BORDER_THICK - 1.0, self.y + 0.5)
        if self.y >= surface_y + APHID_DROP_MIN_DEPTH and self.is_grounded(world):
            self.milk_stock = min(APHID_MAX_MILK_STOCK, self.milk_stock + 0.005)
        if random.random() < 0.01:
            nx = self.x + random.uniform(-0.4, 0.4)
            nx = max(BORDER_THICK, min(GW - BORDER_THICK - 1, nx))
            self.x = nx

    def is_grounded(self, world):
        ix, iy = int(self.x), int(self.y)
        below = iy + 1
        if below >= GH:
            return True
        return world[below, ix] in SOLID_TILES
