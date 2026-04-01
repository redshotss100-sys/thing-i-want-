import math
import random
from collections import defaultdict

import numpy as np

from brain import N_IN, N_OUT, forward
from config import (
    AIR,
    APHID_BODY,
    APHID_DROP_MIN_DEPTH,
    APHID_FARM_BONUS,
    APHID_MAX_POP,
    BORDER,
    BORDER_KILL_ZONE,
    BORDER_THICK,
    CARRYABLE,
    CD_ACTION,
    CD_DIG,
    CD_DROP,
    CD_PICKUP,
    CLIMB_SPEED,
    CONNECT_TUNNEL_BONUS,
    DEPTH_DIG_BONUS,
    DIRT,
    FOOD,
    GH,
    GRASS,
    GRAVITY,
    GROUND_FRICTION,
    GW,
    HORIZ_ACCEL,
    LOOP_DIG_MEMORY,
    LOOP_DIG_PENALTY,
    LOOP_DROP_DIST,
    LOOP_DROP_PENALTY,
    MAX_CARRY,
    MAX_FALL_SPEED,
    MAX_GEN_TICKS,
    MAX_MOVE_SPEED,
    NEST,
    NEST_DIG_BONUS,
    NEST_DROP_BONUS,
    NEST_FOOD_BONUS,
    NEST_PROX_MULT,
    NEST_RADIUS,
    REDIG_FITNESS_MULT,
    ROCK,
    SOLID_TILES,
    TILE_NAMES,
    TUNNEL,
    AIR_FRICTION,
)
from entities import Aphid
from physics import is_grounded, move_with_collision, touching_wall
from world import RoomDetector


class Ant:
    def __init__(self, x, y, weights):
        self.x, self.y = float(x), float(y)
        self.vx, self.vy = 0.0, 0.0
        self.weights = weights
        self.energy = 150.0
        self.alive = True
        self.fitness = 0.0
        self.age = 0
        self.stack = []
        self.on_wall = False
        self.underground = False
        self.cd_pickup = 0
        self.cd_drop = 0
        self.cd_dig = 0
        self.cd_action = 0
        self.tiles_dug = 0
        self.tiles_carried = defaultdict(int)
        self.dist_carried = 0.0
        self.wall_ticks = 0
        self.underground_ticks = 0
        self._last_x = x
        self.milk_count_this_gen = 0
        self.energy_gained = 0.0
        self.death_cause = None
        self._dig_history = []
        self._loop_penalties = 0
        self._loop_events = []
        self._last_nn_out = np.zeros(N_OUT, dtype=np.float32)
        self._ant_id = id(self)

    def sense(self, world, phero, nest_x, nest_y, sim=None):
        ix, iy = int(self.x), int(self.y)
        inputs = []
        for ddy in range(-1, 2):
            for ddx in range(-1, 2):
                nx2, ny2 = ix + ddx, iy + ddy
                tile = world[ny2, nx2] if (0 <= nx2 < GW and 0 <= ny2 < GH) else BORDER
                oh = [0.0] * 10
                oh[min(tile, 9)] = 1.0
                inputs.extend(oh)
        fd = float(phero.food[iy, ix]) if (0 <= iy < GH and 0 <= ix < GW) else 0.0
        dg = float(phero.danger[iy, ix]) if (0 <= iy < GH and 0 <= ix < GW) else 0.0
        hm = float(phero.home[iy, ix]) if (0 <= iy < GH and 0 <= ix < GW) else 0.0
        inputs.extend([fd, dg, hm])
        if 1 <= ix < GW - 1 and 1 <= iy < GH - 1:
            food_gx = float(phero.food[iy, ix + 1]) - float(phero.food[iy, ix - 1])
            food_gy = float(phero.food[iy + 1, ix]) - float(phero.food[iy - 1, ix])
            home_gx = float(phero.home[iy, ix + 1]) - float(phero.home[iy, ix - 1])
            home_gy = float(phero.home[iy + 1, ix]) - float(phero.home[iy - 1, ix])
        else:
            food_gx = food_gy = home_gx = home_gy = 0.0
        inputs.extend([
            np.clip(food_gx, -1, 1), np.clip(food_gy, -1, 1),
            np.clip(home_gx, -1, 1), np.clip(home_gy, -1, 1),
        ])
        inputs.append(self.energy / 200.0)
        inputs.append(len(self.stack) / MAX_CARRY)
        inputs.append(1.0 if self.underground else 0.0)
        inputs.append(1.0 if any(t == FOOD for t in self.stack) else 0.0)
        dx_nest = (nest_x - self.x) / (GW * 0.5)
        dy_nest = (nest_y - self.y) / (GH * 0.5)
        inputs.extend([np.clip(dx_nest, -1, 1), np.clip(dy_nest, -1, 1)])
        dist_nest = math.hypot(self.x - nest_x, self.y - nest_y)
        inputs.append(1.0 if dist_nest < NEST_RADIUS else 0.0)
        nearby = 0
        if sim is not None:
            for other in sim.ants:
                if other is not self and other.alive and math.hypot(other.x - self.x, other.y - self.y) < 8:
                    nearby += 1
        inputs.append(min(nearby, 10) / 10.0)
        avg_signal = 0.0
        if sim is not None and nearby > 0:
            signals = [
                other._last_nn_out[6] for other in sim.ants
                if other is not self and other.alive and math.hypot(other.x - self.x, other.y - self.y) < 8
            ]
            if signals:
                avg_signal = float(np.mean(signals))
        inputs.append(avg_signal)
        room = float(sim.room_detector.room_at(self.x, self.y)) if hasattr(sim, 'room_detector') else 0.0
        inputs.append(room / 5.0)
        spd = math.hypot(self.vx, self.vy) + 0.001
        inputs.extend([np.clip(self.vx / spd, -1, 1), np.clip(self.vy / spd, -1, 1)])
        depth = max(0.0, self.y - (sim.surface_y if sim else GH * 0.25)) / (GH * 0.75)
        inputs.append(min(depth, 1.0))
        arr = np.array(inputs[:N_IN], dtype=np.float32)
        if len(arr) < N_IN:
            arr = np.pad(arr, (0, N_IN - len(arr)))
        return arr

    def _speed_mult(self):
        if not self.stack:
            return 1.0
        m = 1.0
        for t in self.stack:
            m *= CARRYABLE.get(t, 0.8)
        return max(0.2, m)

    def _try_pick_up(self, world, aphids, stats):
        if self.cd_pickup > 0 or len(self.stack) >= MAX_CARRY:
            return
        ix, iy = int(self.x), int(self.y)
        tile = world[iy, ix] if (0 <= ix < GW and 0 <= iy < GH) else AIR
        if tile in CARRYABLE and tile not in (BORDER, NEST):
            world[iy, ix] = TUNNEL
            self.stack.append(tile)
            self.tiles_carried[tile] += 1
            stats['tiles_picked_up'] += 1
            self.fitness += 2
            self.cd_pickup = CD_PICKUP
            return
        for aphid in aphids:
            if aphid.alive and math.hypot(self.x - aphid.x, self.y - aphid.y) < 2:
                aphid.alive = False
                self.stack.append(APHID_BODY)
                self.tiles_carried[APHID_BODY] += 1
                stats['tiles_picked_up'] += 1
                self.fitness += 5
                self.cd_pickup = CD_PICKUP
                return

    def _try_drop(self, world, aphids, nest_x, nest_y, stats, surface_y):
        if self.cd_drop > 0 or not self.stack:
            return
        ix, iy = int(self.x), int(self.y)
        near_nest = math.hypot(self.x - nest_x, self.y - nest_y) < NEST_RADIUS
        for cx, cy in [(ix, iy), (ix, iy - 1), (ix - 1, iy), (ix + 1, iy), (ix, iy + 1)]:
            if BORDER_THICK <= cx < GW - BORDER_THICK and BORDER_THICK <= cy < GH - BORDER_THICK and world[cy, cx] in (AIR, TUNNEL):
                tile = self.stack.pop()
                for dig_x, dig_y, _ in self._dig_history:
                    if math.hypot(cx - dig_x, cy - dig_y) < LOOP_DROP_DIST:
                        self.fitness -= LOOP_DROP_PENALTY
                        stats['drop_loops'] += 1
                        self._loop_penalties += 1
                        break
                if tile == APHID_BODY:
                    grounded = (cy + 1 < GH and world[cy + 1, cx] in SOLID_TILES)
                    underground = cy >= surface_y + APHID_DROP_MIN_DEPTH
                    can_spawn = grounded and underground and len(aphids) < APHID_MAX_POP
                    if can_spawn:
                        aphids.append(Aphid(cx, cy))
                        self.fitness += APHID_FARM_BONUS
                        stats['aphids_farmed'] += 1
                    else:
                        world[cy, cx] = FOOD
                else:
                    world[cy, cx] = tile
                    if tile == FOOD and near_nest:
                        self.fitness += NEST_DROP_BONUS
                        stats['food_delivered'] += 1
                stats['tiles_dropped'] += 1
                self.fitness += 3.0 * (NEST_PROX_MULT if near_nest else 1.0)
                self.cd_drop = CD_DROP
                return

    def _displace_dirt(self, world, dx, dy, nest_x, nest_y, sim):
        is_redig = False
        if hasattr(sim, 'tunnel_history') and sim.tunnel_history[dy, dx] > 0:
            is_redig = True
            self.fitness -= LOOP_DIG_PENALTY
            sim.stats['dig_loops'] += 1
        if hasattr(sim, 'tunnel_history'):
            sim.tunnel_history[dy, dx] = min(255, sim.tunnel_history[dy, dx] + 1)
        self._dig_history.append((dx, dy, sim._gen_tick))
        if len(self._dig_history) > LOOP_DIG_MEMORY:
            self._dig_history.pop(0)
        for cx, cy in [(dx, dy - 1), (dx - 1, dy), (dx + 1, dy), (dx - 1, dy - 1), (dx + 1, dy - 1), (dx, dy - 2), (dx - 1, dy - 2), (dx + 1, dy - 2)]:
            if BORDER_THICK <= cx < GW - BORDER_THICK and BORDER_THICK <= cy < GH - BORDER_THICK and world[cy, cx] in (AIR, TUNNEL):
                world[cy, cx] = DIRT
                world[dy, dx] = TUNNEL
                self.tiles_dug += 1
                sim.stats['dug'] += 1
                for ddx2, ddy2 in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ax2, ay2 = dx + ddx2, dy + ddy2
                    if 0 <= ax2 < GW and 0 <= ay2 < GH and world[ay2, ax2] == TUNNEL and (ax2, ay2) != (cx, cy):
                        self.fitness += CONNECT_TUNNEL_BONUS
                        break
                depth_below_surface = max(0, dy - (GH * 0.30))
                if depth_below_surface > 0:
                    self.fitness += DEPTH_DIG_BONUS * (depth_below_surface // 40)
                if math.hypot(dx - nest_x, dy - nest_y) < NEST_RADIUS:
                    self.fitness += NEST_DIG_BONUS
                    sim.stats['tunnels_near_nest'] += 1
                return is_redig
        return False

    def _check_cavein(self, world, rx, ry, stats):
        count = sum(1 for ddy in range(-2, 3) for ddx in range(-2, 3) if (0 <= rx + ddx < GW and 0 <= ry + ddy < GH and world[ry + ddy, rx + ddx] == TUNNEL))
        if count > 14 and random.random() < 0.04:
            for ddy in range(-3, 4):
                for ddx in range(-3, 4):
                    nx2, ny2 = rx + ddx, ry + ddy
                    if 0 <= nx2 < GW and 0 <= ny2 < GH and world[ny2, nx2] == TUNNEL and random.random() < 0.5:
                        world[ny2, nx2] = DIRT
            if random.random() < 0.6:
                self.death_cause = 'cavein'
                self.alive = False
                stats['deaths_cavein'] += 1
            stats['caveins'] += 1
            age_frac = min(1.0, self.age / max(1, MAX_GEN_TICKS))
            self.fitness -= 20 + 80 * (1.0 - age_frac)

    def update(self, world, phero, aphids, nest_x, nest_y, sim, nn_output=None):
        if not self.alive:
            return
        self.age += 1
        self.underground = (self.y >= sim.surface_y - 1)
        if self.underground:
            self.underground_ticks += 1
        self.cd_pickup = max(0, self.cd_pickup - 1)
        self.cd_drop = max(0, self.cd_drop - 1)
        self.cd_dig = max(0, self.cd_dig - 1)
        self.cd_action = max(0, self.cd_action - 1)
        out = nn_output if nn_output is not None else forward(self.weights, self.sense(world, phero, nest_x, nest_y, sim))
        self._last_nn_out = out.copy()
        move_l, move_r, move_u, dig, pick_up, drop_it, _signal = out

        spd = self._speed_mult()
        max_spd = MAX_MOVE_SPEED * spd
        horiz_intent = move_r - move_l
        self.vx += np.clip(horiz_intent, -1.0, 1.0) * HORIZ_ACCEL * spd
        self.vx = float(np.clip(self.vx, -max_spd, max_spd))
        grounded = is_grounded(world, self.x, self.y)
        self.on_wall = self.underground and touching_wall(world, self.x, self.y)
        if self.on_wall:
            self.wall_ticks += 1
        can_climb = self.on_wall and move_u > 0.2 and self.y >= sim.surface_y
        if can_climb:
            self.vy = min(self.vy, -CLIMB_SPEED * spd)
        self.vy += GRAVITY
        self.vy = float(np.clip(self.vy, -2.2, MAX_FALL_SPEED))
        self.vx *= GROUND_FRICTION if grounded else AIR_FRICTION
        move_with_collision(self, world)

        dig_target_x = int(self.x + (1 if horiz_intent >= 0 else -1))
        dig_target_y = int(self.y - 1) if can_climb and self.vy < 0 else int(self.y)
        if self.cd_dig <= 0 and dig > 0.2 and BORDER_THICK <= dig_target_x < GW - BORDER_THICK and BORDER_THICK <= dig_target_y < GH - BORDER_THICK:
            t = world[dig_target_y, dig_target_x]
            if t in (DIRT, GRASS):
                is_redig = self._displace_dirt(world, dig_target_x, dig_target_y, nest_x, nest_y, sim)
                self.energy += 2.0
                dist_n = math.hypot(dig_target_x - nest_x, dig_target_y - nest_y)
                nm = NEST_PROX_MULT if dist_n < NEST_RADIUS else 1.0
                rm = 1.0
                if hasattr(sim, 'room_detector'):
                    rt = sim.room_detector.room_at(dig_target_x, dig_target_y)
                    rm = 2.0 if rt == RoomDetector.R_NEST else (1.3 if rt == RoomDetector.R_CHAMBER else (1.4 if rt == RoomDetector.R_FARM else 1.0))
                dig_fit = 3.0 * nm * rm
                if is_redig:
                    dig_fit *= REDIG_FITNESS_MULT
                self.fitness += dig_fit
                self.cd_dig = CD_DIG
                self.cd_action = CD_ACTION
            elif t == ROCK:
                self._check_cavein(world, dig_target_x, dig_target_y, sim.stats)

        self.x = max(0.0, min(GW - 1.0, self.x))
        self.y = max(0.0, min(GH - 1.0, self.y))
        ix, iy = int(self.x), int(self.y)
        kz = BORDER_KILL_ZONE
        if self.x < kz or self.x > GW - 1 - kz or self.y < kz or self.y > GH - 1 - kz:
            for tile in self.stack:
                world[min(GH - 1, max(0, iy)), min(GW - 1, max(0, ix))] = tile
            self.stack = []
            self.death_cause = 'border'
            self.alive = False
            sim.stats['deaths_border'] += 1
            age_frac = min(1.0, self.age / max(1, MAX_GEN_TICKS))
            self.fitness -= 50 + 200 * (1.0 - age_frac)
            phero.deposit(self.x, self.y, 'danger', 1.0)
            return

        if self.stack:
            self.dist_carried += abs(self.x - self._last_x)
        self._last_x = self.x

        if self.cd_action <= 0:
            if pick_up > 0.4:
                self._try_pick_up(world, aphids, sim.stats)
                self.cd_action = CD_ACTION
            elif drop_it > 0.4:
                self._try_drop(world, aphids, nest_x, nest_y, sim.stats, sim.surface_y)
                self.cd_action = CD_ACTION

        dist_nest = math.hypot(self.x - nest_x, self.y - nest_y)
        near_nest = dist_nest < NEST_RADIUS
        nest_mult = NEST_PROX_MULT if near_nest else 1.0
        room_mult = 1.0
        if hasattr(sim, 'room_detector'):
            rt = sim.room_detector.room_at(self.x, self.y)
            room_mult = 2.0 if rt == RoomDetector.R_NEST else (1.5 if rt == RoomDetector.R_FARM else (1.3 if rt == RoomDetector.R_CHAMBER else 1.0))

        if self.stack and self.stack[-1] == FOOD and self.energy < 160:
            self.stack.pop()
            gained = 60.0
            self.energy = min(200, self.energy + gained)
            self.energy_gained += gained
            self.fitness += (20.0 + NEST_FOOD_BONUS) * nest_mult * room_mult if near_nest else 20.0 * room_mult
            if near_nest:
                sim.stats['food_delivered'] += 1
            phero.deposit(self.x, self.y, 'food', 0.6)
            sim.stats['food_eaten'] += 1
        elif 0 <= ix < GW and 0 <= iy < GH and world[iy, ix] == FOOD:
            world[iy, ix] = TUNNEL
            gained = 60.0
            self.energy = min(200, self.energy + gained)
            self.energy_gained += gained
            self.fitness += (30.0 + NEST_FOOD_BONUS) * nest_mult * room_mult if near_nest else 30.0 * room_mult
            if near_nest:
                sim.stats['food_delivered'] += 1
            phero.deposit(self.x, self.y, 'food', 0.8)
            sim.stats['food_eaten'] += 1

        for aphid in aphids:
            if not aphid.alive or not aphid.is_grounded(world) or self.y < sim.surface_y - 2:
                continue
            if math.hypot(self.x - aphid.x, self.y - aphid.y) >= 3:
                continue
            if drop_it > 0.5 and pick_up <= 0.5:
                aphid.alive = False
                ax, ay = int(aphid.x), int(aphid.y)
                if 0 <= ax < GW and 0 <= ay < GH:
                    world[ay, ax] = FOOD
                self.fitness += 20.0 * nest_mult * room_mult
                sim.stats['aphids_killed'] += 1
            elif pick_up > 0.3 and self.cd_action <= 0 and aphid.can_milk():
                self.milk_count_this_gen += 1
                diminish = max(0.1, 1.0 - self.milk_count_this_gen * 0.02)
                gained = 25.0 * diminish
                aphid.get_milked()
                self.energy = min(200, self.energy + gained)
                self.energy_gained += gained
                self.fitness += max(1, 15 * diminish) * nest_mult * room_mult
                phero.deposit(self.x, self.y, 'food', 0.4)
                sim.stats['aphids_milked'] += 1
                self.cd_action = CD_ACTION

        self.energy -= 0.10 + 0.025 * len(self.stack)
        if self.energy <= 0:
            for tile in self.stack:
                ddx = int(self.x) + random.randint(-2, 2)
                ddy = int(self.y) + random.randint(-2, 2)
                if BORDER_THICK <= ddx < GW - BORDER_THICK and BORDER_THICK <= ddy < GH - BORDER_THICK and world[ddy, ddx] in (AIR, TUNNEL):
                    world[ddy, ddx] = tile
            self.stack = []
            self.death_cause = 'starvation'
            self.alive = False
            sim.stats['deaths_starvation'] += 1
            age_frac = min(1.0, self.age / max(1, MAX_GEN_TICKS))
            self.fitness -= 30 + 120 * (1.0 - age_frac)
            phero.deposit(self.x, self.y, 'danger', 0.7)
            return

        if 1 <= ix < GW - 1 and 1 <= iy < GH - 1:
            gx = float(phero.food[iy, ix + 1]) - float(phero.food[iy, ix - 1])
            gy = float(phero.food[iy + 1, ix]) - float(phero.food[iy - 1, ix])
            self.vx += gx * 0.30
            self.vy += gy * 0.20
        if self.energy > 40:
            phero.deposit(self.x, self.y, 'food', 0.02)
        if near_nest:
            phero.deposit(self.x, self.y, 'home', 0.08)
        elif self.energy > 60:
            phero.deposit(self.x, self.y, 'home', 0.01)
