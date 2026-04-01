import datetime
import json
import os
import pickle
import random
import sys
import traceback

import numpy as np

try:
    import pygame  # type: ignore
except Exception:
    pygame = None

from ant import Ant
from brain import batch_forward, crossover, mutate, random_brain
from config import *
from entities import Aphid, Spider
from save_system import StatsLogger, load_simulation, new_save_dir, write_crash_dump
from world import Pheromones, RoomDetector

if pygame is not None:
    from render import draw_ant_inspection, draw_ui, find_ant_at, render_world, screen_to_world
else:
    draw_ant_inspection = draw_ui = find_ant_at = render_world = screen_to_world = None


class Simulation:
    def __init__(self, screen, save_dir, gene_pool=None, start_gen=1):
        self.screen = screen
        self.save_dir = save_dir
        self.speed = 1
        self.gen = start_gen
        self.phero = Pheromones()
        self.logger = StatsLogger(save_dir)
        self.stats = self._blank_stats()
        self.best_weights = []
        self.gene_pool = gene_pool or [random_brain() for _ in range(COLONY_SIZE)]
        self._fitness_history = []
        self._stagnation_count = 0
        self._mut_rate = 0.08
        self._mut_noise = 0.15
        self._all_ants_this_gen = []
        self._gen_tick = 0
        self._pool_snapshots = []
        self._exploit_rollback = False
        self._last_death_summary = {}
        self.room_detector = RoomDetector()
        self.tunnel_history = np.zeros((GH, GW), dtype=np.uint8)
        self.camera_zoom = 1.0
        self.camera_x = BASE_W / 2.0
        self.camera_y = BASE_H / 2.0
        self.camera_dragging = False
        self.camera_drag_start = (0, 0)
        self.camera_drag_cam_start = (0.0, 0.0)
        self.selected_ant = None
        self.hovered_ant = None
        self.fullscreen = False
        self._headless_last_spawn_ms = 0.0
        self._visual_enabled = pygame is not None and screen is not None
        if self._visual_enabled:
            self.surface = pygame.Surface((BASE_W, BASE_H))
            self.pixel_array = np.zeros((GH, GW, 3), dtype=np.uint8)
            self._scaled = np.zeros((BASE_H, BASE_W, 3), dtype=np.uint8)
            self._font_ui = pygame.font.SysFont("Consolas", 13)
            self._font_flash = pygame.font.SysFont("Consolas", 18)
        else:
            self.surface = None
            self.pixel_array = None
            self._scaled = None
            self._font_ui = None
            self._font_flash = None
        self._build_world()

    def _blank_stats(self):
        return {k: 0 for k in (
            'dug', 'food_eaten', 'food_delivered', 'aphids_killed', 'aphids_milked', 'aphids_farmed',
            'caveins', 'spider_kills', 'tiles_picked_up', 'tiles_dropped', 'tunnels_near_nest',
            'dig_loops', 'drop_loops', 'deaths_border', 'deaths_starvation', 'deaths_spider', 'deaths_cavein'
        )}

    def _build_world(self):
        self.world = np.zeros((GH, GH if False else GW), dtype=np.uint8)
        self.surface_y = int(GH * 0.25)
        self.world[self.surface_y - 2:self.surface_y, :] = GRASS
        self.world[self.surface_y:, :] = DIRT
        for _ in range(25):
            rx = random.randint(BORDER_THICK + 2, GW - BORDER_THICK - 10)
            ry = random.randint(int(GH * 0.55), GH - BORDER_THICK - 8)
            rh, rw = random.randint(3, 7), random.randint(4, 10)
            self.world[ry:ry + rh, rx:rx + rw] = ROCK
        chamber_w, chamber_h = 14, 10
        chamber_x = GW // 2 - chamber_w // 2
        chamber_y = self.surface_y + 4
        for dy in range(chamber_h):
            for dx in range(chamber_w):
                cy2, cx2 = chamber_y + dy, chamber_x + dx
                if BORDER_THICK <= cx2 < GW - BORDER_THICK and BORDER_THICK <= cy2 < GH - BORDER_THICK:
                    self.world[cy2, cx2] = TUNNEL
        self.entrance_x = GW // 2 - 1
        for dy in range(self.surface_y - 2, chamber_y):
            for dx in range(2):
                ex = self.entrance_x + dx
                if BORDER_THICK <= ex < GW - BORDER_THICK and BORDER_THICK <= dy < GH - BORDER_THICK:
                    self.world[dy, ex] = TUNNEL
        self.nest_x, self.nest_y = GW // 2, chamber_y + chamber_h // 2
        for dy in range(-1, 2):
            for dx in range(-3, 4):
                ny2, nx2 = self.nest_y + dy, self.nest_x + dx
                if 0 <= nx2 < GW and 0 <= ny2 < GH:
                    self.world[ny2, nx2] = NEST
        star_rng = np.random.default_rng(42)
        self._star_x = star_rng.integers(0, GW, 120)
        self._star_y = star_rng.integers(0, max(1, self.surface_y), 120)
        self.aphids = [Aphid(random.randint(BORDER_THICK, GW - BORDER_THICK - 1), self.surface_y - 2) for _ in range(20)]
        self.spiders = [Spider(random.randint(BORDER_THICK, GW - BORDER_THICK - 1), random.randint(int(GH * 0.45), int(GH * 0.65))) for _ in range(3)]
        self.ants, self._all_ants_this_gen = [], []
        for w in self.gene_pool:
            spawn_x = max(BORDER_KILL_ZONE + 1.0, min(GW - BORDER_KILL_ZONE - 2.0, float(self.entrance_x + random.randint(-8, 10))))
            ant = Ant(spawn_x, float(self.surface_y - 3), w)
            self.ants.append(ant)
            self._all_ants_this_gen.append(ant)
        self.stats = self._blank_stats()
        self._gen_tick = 0
        self._headless_last_spawn_ms = 0.0
        self.tunnel_history[:] = 0
        self.room_detector.detect(self.world, self.surface_y)
        self._stamp_border()

    def _stamp_border(self):
        for i in range(BORDER_THICK):
            self.world[i, :], self.world[GH - 1 - i, :], self.world[:, i], self.world[:, GW - 1 - i] = BORDER, BORDER, BORDER, BORDER

    def _check_exploits(self):
        flags = []
        if self.stats['aphids_milked'] > EXPLOIT_MILK_PER_GEN:
            flags.append(f"milk_exploit:{self.stats['aphids_milked']}")
        for ant in self.ants:
            if ant.energy_gained > ant.age * EXPLOIT_ENERGY_GAIN:
                flags.append(f"energy_exploit:ant_age{ant.age}_gained{ant.energy_gained:.0f}")
        return flags

    def _do_rollback(self, steps_back):
        if len(self._pool_snapshots) < steps_back:
            steps_back = len(self._pool_snapshots)
        if steps_back == 0:
            return
        snap_gen, snap_pool = self._pool_snapshots[-steps_back]
        self.gene_pool = [list(w) for w in snap_pool]
        print(f"[ROLLBACK] Reverted to gen {snap_gen} gene pool")

    def _evolve(self):
        exploit_flags = self._check_exploits()
        self.stats['rooms_nest'] = self.room_detector.counts.get(RoomDetector.R_NEST, 0)
        self.stats['rooms_farm'] = self.room_detector.counts.get(RoomDetector.R_FARM, 0)
        self.stats['rooms_chamber'] = self.room_detector.counts.get(RoomDetector.R_CHAMBER, 0)
        self.logger.log_generation(self.gen, self._all_ants_this_gen, self.stats, exploit_flags)
        self.logger.write_summary()
        if exploit_flags:
            steps = 6 if self._exploit_rollback else 3
            self._do_rollback(steps)
            self._exploit_rollback = True
            print(f"[EXPLOIT] Flags: {exploit_flags}  Rolling back {steps} gens.")
        else:
            self._exploit_rollback = False
            ants = self._all_ants_this_gen
            if ants:
                colony_bonus = self.stats.get('food_delivered', 0) * 15.0 + self.stats.get('dug', 0) * 0.5 + self.stats.get('aphids_milked', 0) * 2.0 + self.stats.get('tunnels_near_nest', 0)
                colony_bonus_per_ant = colony_bonus / max(1, len(ants))
                for ant in ants:
                    ant.fitness = ant.fitness * (1.0 - COLONY_FITNESS_WEIGHT) + colony_bonus_per_ant * COLONY_FITNESS_WEIGHT
                avg_fit = float(np.mean([a.fitness for a in ants]))
                self._fitness_history.append(avg_fit)
                if len(self._fitness_history) > 40:
                    self._fitness_history.pop(0)
                improved = len(self._fitness_history) < 2 or avg_fit > max(self._fitness_history[:-1]) * 1.005
                if improved:
                    self._stagnation_count, self._mut_rate, self._mut_noise = 0, 0.08, 0.15
                else:
                    self._stagnation_count += 1
                if self._stagnation_count >= STAGNATION_GENS:
                    self._mut_rate, self._mut_noise = STAGNATION_RATE, STAGNATION_NOISE
                survivors = sorted(ants, key=lambda a: a.fitness, reverse=True)
                elite = [a.weights for a in survivors[:max(5, len(survivors) // 4)]]
                self.best_weights = elite[:5]
                new_pool = list(elite)
                while len(new_pool) < COLONY_SIZE:
                    p1, p2 = random.choices(elite, k=2)
                    new_pool.append(mutate(crossover(p1, p2), rate=self._mut_rate, noise=self._mut_noise))
                self.gene_pool = new_pool
        self._last_death_summary = {
            'border': self.stats.get('deaths_border', 0), 'starvation': self.stats.get('deaths_starvation', 0),
            'spider': self.stats.get('deaths_spider', 0), 'cavein': self.stats.get('deaths_cavein', 0),
            'looping': self.stats.get('deaths_looping', 0),
        }
        if self.gen % 5 == 0:
            self._pool_snapshots.append((self.gen, [list(w) for w in self.gene_pool]))
            if len(self._pool_snapshots) > 10:
                self._pool_snapshots.pop(0)
        if self.gen % AUTOSAVE_INTERVAL == 0:
            print(f"[AUTOSAVE gen {self.gen}] {'OK' if self.save() else 'FAILED'}")

    def _spread_grass(self):
        if random.random() < 0.04:
            gy, gx = np.where(self.world == GRASS)
            if len(gx) == 0:
                return
            idx = random.randint(0, len(gx) - 1)
            ox, oy = gx[idx] + random.randint(-2, 2), gy[idx] + random.randint(-1, 1)
            if BORDER_THICK <= ox < GW - BORDER_THICK and BORDER_THICK <= oy < GH - BORDER_THICK and self.world[oy, ox] == AIR:
                self.world[oy, ox] = GRASS
            if random.random() < 0.002 and len(self.aphids) < 40:
                self.aphids.append(Aphid(random.randint(BORDER_THICK, GW - BORDER_THICK - 1), self.surface_y - 2))

    def _colony_collapse_reason(self):
        if self._gen_tick < FPS * 90:
            return None
        no_progress = self.stats['dug'] == 0 and self.stats['food_eaten'] == 0 and self.stats['food_delivered'] == 0
        if no_progress:
            return "no_progress"
        if self.stats.get('deaths_border', 0) > int(0.70 * COLONY_SIZE):
            return "border_abuse"
        return None

    def save(self):
        try:
            np.save(os.path.join(self.save_dir, "world.npy"), self.world)
            np.save(os.path.join(self.save_dir, "phero_food.npy"), self.phero.food)
            np.save(os.path.join(self.save_dir, "phero_danger.npy"), self.phero.danger)
            np.save(os.path.join(self.save_dir, "phero_home.npy"), self.phero.home)
            np.save(os.path.join(self.save_dir, "room_grid.npy"), self.room_detector.grid)
            np.save(os.path.join(self.save_dir, "tunnel_history.npy"), self.tunnel_history)
            with open(os.path.join(self.save_dir, "brains.pkl"), "wb") as f:
                pickle.dump({"gene_pool": self.gene_pool, "best_weights": self.best_weights, "gen": self.gen, "snapshots": self._pool_snapshots}, f)
            meta = {
                "gen": self.gen, "speed": self.speed, "surface_y": self.surface_y, "nest_x": self.nest_x,
                "nest_y": self.nest_y, "colony_size": COLONY_SIZE, "saved_at": datetime.datetime.now().isoformat(),
                "aphids": [[a.x, a.y, a.milk_cooldown, a.milk_stock] for a in self.aphids if a.alive],
                "spiders": [[s.x, s.y, s.vx, s.vy] for s in self.spiders if s.alive],
                "stats": self.stats, "room_counts": dict(self.room_detector.counts),
            }
            with open(os.path.join(self.save_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            self.logger.write_summary()
            return True
        except Exception:
            write_crash_dump(sys.exc_info(), "Error in save()")
            return False

    def _flash(self, msg, ms=2500):
        if not self._visual_enabled:
            print(msg)
            return
        surf = self._font_flash.render(msg, True, (255, 255, 80))
        start = pygame.time.get_ticks()
        while pygame.time.get_ticks() - start < ms:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return
            ww, wh = self.screen.get_size()
            self.screen.blit(surf, (ww // 2 - surf.get_width() // 2, wh // 2 - surf.get_height() // 2))
            pygame.display.flip()

    def _generation_end(self, reason):
        print(f"[GEN {self.gen} END] reason={reason}  survivors={len(self.ants)}/{COLONY_SIZE}  ticks={self._gen_tick}")
        self._evolve()
        self.gen += 1
        self._build_world()
        self.phero.decay()
        return reason

    def step(self, current_time_ms=None):
        self._gen_tick += 1
        alive_ants = self.ants
        outputs = None
        if alive_ants:
            inputs = np.array([a.sense(self.world, self.phero, self.nest_x, self.nest_y, self) for a in alive_ants], dtype=np.float32)
            outputs = batch_forward([a.weights for a in alive_ants], inputs)
        for idx, ant in enumerate(list(self.ants)):
            ant.update(self.world, self.phero, self.aphids, self.nest_x, self.nest_y, self, nn_output=outputs[idx] if alive_ants else None)
        self.ants = [a for a in self.ants if a.alive]
        for aphid in self.aphids:
            aphid.update(self.world, self.surface_y)
        self.aphids = [a for a in self.aphids if a.alive]
        for sp in self.spiders:
            sp.update(self.world, self.ants, self.phero, self.stats)
        self.spiders = [s for s in self.spiders if s.alive]
        if len(self.spiders) < 2 and random.random() < 0.003:
            self.spiders.append(Spider(random.randint(BORDER_THICK, GW - BORDER_THICK - 1), random.randint(int(GH * 0.45), int(GH * 0.65))))
        if current_time_ms is None:
            if pygame is None:
                current_time_ms = self._gen_tick * (1000.0 / FPS)
            else:
                current_time_ms = pygame.time.get_ticks()
        if current_time_ms - self._headless_last_spawn_ms >= 60000 and len(self.ants) < COLONY_SIZE and self.best_weights:
            w = mutate(random.choice(self.best_weights), rate=0.05, noise=0.08)
            spawn_x = max(BORDER_KILL_ZONE + 1.0, min(GW - BORDER_KILL_ZONE - 2.0, float(self.entrance_x + random.randint(-8, 10))))
            new_ant = Ant(spawn_x, float(self.surface_y - 3), w)
            self.ants.append(new_ant)
            self._all_ants_this_gen.append(new_ant)
            self._headless_last_spawn_ms = current_time_ms
        self._spread_grass()
        self._stamp_border()
        if self._gen_tick % ROOM_DETECT_INTERVAL == 0:
            self.room_detector.detect(self.world, self.surface_y)
        if self._gen_tick % 3 == 0:
            self.phero.decay()
            self.phero.diffuse()
        early_wipe = len(self.ants) < MIN_SURVIVORS and len(self._all_ants_this_gen) > 0 and len(self.ants) < len(self._all_ants_this_gen)
        time_limit = self._gen_tick >= MAX_GEN_TICKS
        collapse_reason = self._colony_collapse_reason()
        collapsed = collapse_reason is not None
        if not self.ants or early_wipe or time_limit or collapsed:
            reason = "colony_dead" if not self.ants else ("early_wipe" if early_wipe else ("time_limit" if time_limit else collapse_reason))
            return self._generation_end(reason)
        return None

    def run(self):
        if pygame is None:
            raise RuntimeError("pygame is not available; use headless.py instead.")
        clock = pygame.time.Clock()
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return
                if e.type == pygame.VIDEORESIZE and not self.fullscreen:
                    self.screen = pygame.display.set_mode((e.w, e.h), pygame.RESIZABLE)
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_q:
                        return
                    if e.key == pygame.K_s:
                        ok = self.save()
                        self._flash("Saved to " + os.path.basename(self.save_dir) if ok else "Save FAILED — see crash dump in saves/")
                    if e.key == pygame.K_r:
                        self.gen = 1
                        self.gene_pool = [random_brain() for _ in range(COLONY_SIZE)]
                        self._pool_snapshots = []
                        self._build_world()
                        self.selected_ant = None
                    if e.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                        self.speed = int(e.unicode)
                    if e.key == pygame.K_f:
                        self.fullscreen = not self.fullscreen
                        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN) if self.fullscreen else pygame.display.set_mode((BASE_W, BASE_H), pygame.RESIZABLE)
                if e.type == pygame.MOUSEWHEEL:
                    mx, my = pygame.mouse.get_pos()
                    wx, wy = screen_to_world(self, mx, my)
                    self.camera_zoom = min(8.0, self.camera_zoom * 1.15) if e.y > 0 else max(1.0, self.camera_zoom / 1.15)
                    nwx, nwy = screen_to_world(self, mx, my)
                    self.camera_x -= (nwx - wx)
                    self.camera_y -= (nwy - wy)
                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 3:
                    self.camera_dragging = True
                    self.camera_drag_start = e.pos
                    self.camera_drag_cam_start = (self.camera_x, self.camera_y)
                    self.selected_ant = None
                if e.type == pygame.MOUSEBUTTONUP and e.button == 3:
                    self.camera_dragging = False
                if e.type == pygame.MOUSEMOTION and self.camera_dragging:
                    dx, dy = e.pos[0] - self.camera_drag_start[0], e.pos[1] - self.camera_drag_start[1]
                    ww, wh = self.screen.get_size()
                    vw, vh = BASE_W / self.camera_zoom, BASE_H / self.camera_zoom
                    self.camera_x = self.camera_drag_cam_start[0] - (dx / max(1, ww)) * vw
                    self.camera_y = self.camera_drag_cam_start[1] - (dy / max(1, wh)) * vh
                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    self.selected_ant = find_ant_at(self, *e.pos)

            if self.selected_ant is not None and self.selected_ant.alive and not self.camera_dragging:
                self.camera_x, self.camera_y = self.selected_ant.x * TILE, self.selected_ant.y * TILE
            elif self.selected_ant is not None and not self.selected_ant.alive:
                self.selected_ant = None
            self.hovered_ant = find_ant_at(self, *pygame.mouse.get_pos())

            for _ in range(self.speed):
                self.step(current_time_ms=pygame.time.get_ticks())

            render_world(self)
            draw_ant_inspection(self)
            draw_ui(self)
            pygame.display.flip()
            clock.tick(FPS)

    def run_headless(self, generations=1, print_every_ticks=600, save_on_exit=True):
        target_gen = self.gen + max(0, generations)
        try:
            while self.gen < target_gen:
                reason = self.step(current_time_ms=self._gen_tick * (1000.0 / FPS))
                if print_every_ticks and self._gen_tick % print_every_ticks == 0:
                    print(
                        f"[GEN {self.gen}] tick={self._gen_tick} ants={len(self.ants)} food={self.stats['food_eaten']} "
                        f"dug={self.stats['dug']} aphids={len(self.aphids)} spiders={len(self.spiders)}"
                    )
                if reason is not None:
                    summary = self.logger.all_gens[-1] if self.logger.all_gens else {}
                    print(
                        f"[HEADLESS] finished gen {self.gen - 1} reason={reason} avg_fit={summary.get('avg_fitness', 0):.2f} "
                        f"avg_life={summary.get('avg_lifespan', 0):.1f} food={summary.get('food_eaten', 0)} dug={summary.get('tiles_dug_this_gen', 0)}"
                    )
            if save_on_exit:
                self.save()
        except Exception:
            dump = write_crash_dump(sys.exc_info(), "Headless runner crash")
            print(f"[HEADLESS CRASH] {dump}")
            raise


def run_menu(screen):
    if pygame is None:
        raise RuntimeError("pygame is not available; use headless.py instead.")
    font_big = pygame.font.SysFont("Consolas", 38, bold=True)
    font_med = pygame.font.SysFont("Consolas", 22)
    font_small = pygame.font.SysFont("Consolas", 13)
    clock = pygame.time.Clock()
    buttons = {
        "new": pygame.Rect(BASE_W // 2 - 140, 300, 280, 55),
        "load": pygame.Rect(BASE_W // 2 - 140, 375, 280, 55),
        "quit": pygame.Rect(BASE_W // 2 - 140, 450, 280, 55),
    }
    labels = {"new": "New Game", "load": "Load Save", "quit": "Quit"}
    colours = {"new": ((50, 110, 50), (70, 180, 70)), "load": ((50, 70, 130), (70, 110, 190)), "quit": ((110, 35, 35), (170, 55, 55))}
    while True:
        mx, my = pygame.mouse.get_pos()
        hovered = next((k for k, r in buttons.items() if r.collidepoint(mx, my)), None)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return 'quit', None
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                if hovered == 'new':
                    return 'new', None
                if hovered == 'load':
                    try:
                        import tkinter as tk
                        from tkinter import filedialog
                        root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
                        chosen = filedialog.askdirectory(title="Select save folder", initialdir=SAVES_ROOT)
                        root.destroy()
                    except Exception:
                        chosen = None
                    if chosen and os.path.isfile(os.path.join(chosen, "meta.json")):
                        return 'load', chosen
                if hovered == 'quit':
                    return 'quit', None
        screen.fill((12, 10, 6))
        title = font_big.render("ANT EVOLUTION SIM", True, (255, 210, 60))
        screen.blit(title, (BASE_W // 2 - title.get_width() // 2, 150))
        sub = font_med.render("generational neural net colony simulator", True, (130, 110, 70))
        screen.blit(sub, (BASE_W // 2 - sub.get_width() // 2, 205))
        for key, rect in buttons.items():
            base, hi = colours[key]
            col = hi if hovered == key else base
            pygame.draw.rect(screen, col, rect, border_radius=8)
            pygame.draw.rect(screen, (180, 180, 180), rect, 1, border_radius=8)
            lbl = font_med.render(labels[key], True, (255, 255, 255))
            screen.blit(lbl, (rect.centerx - lbl.get_width() // 2, rect.centery - lbl.get_height() // 2))
        hint = font_small.render(f"saves: {SAVES_ROOT}", True, (70, 70, 70))
        screen.blit(hint, (BASE_W // 2 - hint.get_width() // 2, BASE_H - 35))
        pygame.display.flip()
        clock.tick(60)


def main():
    if pygame is None:
        raise RuntimeError("pygame is not installed. Run headless.py for no-window testing.")
    pygame.init()
    screen = pygame.display.set_mode((BASE_W, BASE_H), pygame.RESIZABLE)
    pygame.display.set_caption("Ant Evolution Sim v4")
    try:
        action, save_dir = run_menu(screen)
        if action == 'quit':
            pygame.quit(); return
        if action == 'new':
            sim = Simulation(screen, new_save_dir())
        else:
            try:
                sim = load_simulation(screen, save_dir)
            except Exception:
                dump = write_crash_dump(sys.exc_info(), f"Failed to load: {save_dir}")
                font = pygame.font.SysFont("Consolas", 15)
                screen.fill((30, 0, 0))
                for i, line in enumerate(["Failed to load save — crash dump written:", dump, "", "Starting new game..."]):
                    screen.blit(font.render(line, True, (255, 100, 100)), (40, 200 + i * 26))
                pygame.display.flip(); pygame.time.wait(3500)
                sim = Simulation(screen, new_save_dir())
        sim.run()
    except Exception:
        dump = write_crash_dump(sys.exc_info(), "Top-level crash")
        try:
            font = pygame.font.SysFont("Consolas", 15)
            screen.fill((20, 0, 0))
            err = traceback.format_exc().strip().split("\n")[-1]
            lines = ["THE SIMULATION CRASHED", "", "Crash dump saved to:", dump, "", err, "", "Press any key or close window to exit."]
            for i, line in enumerate(lines):
                screen.blit(font.render(line, True, (255, 80, 80)), (40, 60 + i * 24))
            pygame.display.flip()
            waiting = True
            while waiting:
                for e in pygame.event.get():
                    if e.type in (pygame.QUIT, pygame.KEYDOWN):
                        waiting = False
                pygame.time.wait(50)
        except Exception:
            pass
    pygame.quit()


if __name__ == "__main__":
    main()
