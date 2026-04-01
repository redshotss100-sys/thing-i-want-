import math

from config import GW, GH, SOLID_TILES


def is_walkable(world, x, y):
    if not (0 <= x < GW and 0 <= y < GH):
        return False
    return world[y, x] not in SOLID_TILES


def is_grounded(world, x, y):
    ix, iy = int(x), int(y)
    by = iy + 1
    if by >= GH:
        return True
    if not (0 <= ix < GW):
        return False
    return world[by, ix] in SOLID_TILES


def touching_wall(world, x, y):
    ix, iy = int(x), int(y)
    checks = ((ix - 1, iy), (ix + 1, iy), (ix - 1, iy + 1), (ix + 1, iy + 1))
    for wx, wy in checks:
        if 0 <= wx < GW and 0 <= wy < GH and world[wy, wx] in SOLID_TILES:
            return True
    return False


def move_with_collision(entity, world):
    next_x = entity.x + entity.vx
    if is_walkable(world, int(next_x), int(entity.y)):
        entity.x = next_x
    else:
        entity.vx = 0.0

    next_y = entity.y + entity.vy
    if is_walkable(world, int(entity.x), int(next_y)):
        entity.y = next_y
    else:
        if entity.vy > 0:
            entity.y = math.floor(entity.y)
        entity.vy = 0.0
