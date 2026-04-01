import datetime
import math
import os

import numpy as np
import pygame

from config import (
    APHID_BODY,
    BASE_H,
    BASE_W,
    COLONY_SIZE,
    DIRT,
    MAX_CARRY,
    MIN_SURVIVORS,
    PALETTE,
    TILE,
    TILE_NAMES,
    WALL_BG,
)
from world import RoomDetector, get_dirt_noise, sky_colour


def render_world(sim):
    arr = sim.pixel_array
    sky = sky_colour()
    sy = sim.surface_y
    arr[:sy, :] = sky
    if sy > 0:
        fade = np.linspace(0.62, 1.0, sy, dtype=np.float32).reshape(sy, 1, 1)
        arr[:sy] = np.clip(arr[:sy].astype(np.float32) * fade, 0, 255).astype(np.uint8)
    arr[sy:, :] = WALL_BG
    for tv, col in PALETTE.items():
        arr[sim.world == tv] = col
    dirt_mask = sim.world == DIRT
    if dirt_mask.any():
        noise = get_dirt_noise()
        for c in range(3):
            channel = arr[:, :, c].astype(np.int16)
            channel[dirt_mask] += noise[:, :, c][dirt_mask]
            arr[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
    food_a = (sim.phero.food * 55).astype(np.uint8)
    danger_a = (sim.phero.danger * 75).astype(np.uint8)
    arr[:, :, 1] = np.clip(arr[:, :, 1].astype(np.int16) + food_a, 0, 255).astype(np.uint8)
    arr[:, :, 0] = np.clip(arr[:, :, 0].astype(np.int16) + danger_a, 0, 255).astype(np.uint8)

    room_colors = {
        RoomDetector.R_TUNNEL: (0, 15, 40),
        RoomDetector.R_NEST: (35, 30, 0),
        RoomDetector.R_FARM: (0, 30, 10),
        RoomDetector.R_STORAGE: (30, 15, 0),
        RoomDetector.R_CHAMBER: (20, 10, 25),
    }
    for rtype, tint in room_colors.items():
        rmask = (sim.room_detector.grid == rtype)
        if rmask.any():
            for c in range(3):
                channel = arr[:, :, c].astype(np.int16)
                channel[rmask] = np.clip(channel[rmask] + tint[c], 0, 255)
                arr[:, :, c] = channel.astype(np.uint8)

    brightness = (sky[0] + sky[1] + sky[2]) / 3
    if brightness < 60 and sy > 0:
        sb = int(200 * (1.0 - brightness / 60))
        for sx2, sy2 in zip(sim._star_x, sim._star_y):
            if sy2 < sy:
                arr[sy2, sx2] = (sb, sb, sb)

    s = sim._scaled
    s[0::2, 0::2] = arr
    s[1::2, 0::2] = arr
    s[0::2, 1::2] = arr
    s[1::2, 1::2] = arr
    pygame.surfarray.blit_array(sim.surface, s.transpose(1, 0, 2))

    for aphid in sim.aphids:
        if aphid.alive:
            pygame.draw.circle(sim.surface, PALETTE[APHID_BODY], (int(aphid.x) * TILE, int(aphid.y) * TILE), 2)
    for sp in sim.spiders:
        if sp.alive:
            pygame.draw.circle(sim.surface, (200, 0, 0), (int(sp.x) * TILE, int(sp.y) * TILE), 3)
    for ant in sim.ants:
        if not ant.alive:
            continue
        ax_px, ay_px = int(ant.x) * TILE, int(ant.y) * TILE
        is_hovered = (ant is sim.hovered_ant)
        is_selected = (ant is sim.selected_ant and ant.alive)
        body_col = (255, 255, 255) if is_hovered else ((255, 200, 100) if is_selected else (255, 230, 50))
        pygame.draw.rect(sim.surface, body_col, (ax_px - TILE, ay_px, TILE * 2, TILE))
        if ant.on_wall:
            pygame.draw.rect(sim.surface, (160, 200, 255), (ax_px - TILE, ay_px, TILE * 2, TILE), 1)
        if is_selected:
            pygame.draw.circle(sim.surface, (100, 255, 100), (ax_px, ay_px), 6, 1)
        for i, tile in enumerate(ant.stack):
            pygame.draw.circle(sim.surface, PALETTE.get(tile, (200, 200, 200)), (ax_px, ay_px - 3 - i * 3), 2)

    ww, wh = sim.screen.get_size()
    vw = int(BASE_W / sim.camera_zoom)
    vh = int(BASE_H / sim.camera_zoom)
    vx = max(0, min(BASE_W - vw, int(sim.camera_x - vw / 2)))
    vy = max(0, min(BASE_H - vh, int(sim.camera_y - vh / 2)))
    vw = max(1, min(vw, BASE_W - vx))
    vh = max(1, min(vh, BASE_H - vy))
    src = sim.surface.subsurface(pygame.Rect(vx, vy, vw, vh))
    sim.screen.blit(pygame.transform.scale(src, (ww, wh)), (0, 0))


def screen_to_world(sim, sx, sy):
    ww, wh = sim.screen.get_size()
    vw = BASE_W / sim.camera_zoom
    vh = BASE_H / sim.camera_zoom
    vx = max(0, min(BASE_W - vw, sim.camera_x - vw / 2))
    vy = max(0, min(BASE_H - vh, sim.camera_y - vh / 2))
    return vx + (sx / max(1, ww)) * vw, vy + (sy / max(1, wh)) * vh


def find_ant_at(sim, sx, sy):
    wx, wy = screen_to_world(sim, sx, sy)
    tx, ty = wx / TILE, wy / TILE
    best = None
    best_d = 8.0
    for ant in sim.ants:
        if not ant.alive:
            continue
        d = math.hypot(ant.x - tx, ant.y - ty)
        if d < best_d:
            best_d = d
            best = ant
    return best


def draw_ant_inspection(sim):
    target = sim.hovered_ant
    if target is None and sim.selected_ant is not None and sim.selected_ant.alive:
        target = sim.selected_ant
    if target is None:
        return
    a = target
    ww, _ = sim.screen.get_size()
    panel_w, panel_h, panel_x, panel_y = 260, 310, ww - 270, 10
    panel_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel_surf.fill((10, 10, 20, 200))
    pygame.draw.rect(panel_surf, (120, 120, 160, 200), (0, 0, panel_w, panel_h), 1)
    sim.screen.blit(panel_surf, (panel_x, panel_y))
    f, y, x, lh = sim._font_ui, panel_y + 8, panel_x + 10, 16
    title = "ANT INSPECTION" + ("  [FOLLOWING]" if a is sim.selected_ant else "")
    sim.screen.blit(f.render(title, True, (255, 220, 80)), (x, y))
    y += lh + 4
    rt = sim.room_detector.room_at(a.x, a.y)
    room_name = RoomDetector.NAMES.get(rt, str(rt)) if rt else "none"
    stat_lines = [
        (f"Age: {a.age}  Energy: {a.energy:.0f}/200", (200, 200, 200)),
        (f"Fitness: {a.fitness:.1f}", (255, 255, 100)),
        (f"Tiles dug: {a.tiles_dug}  Carried: {sum(a.tiles_carried.values())}", (200, 200, 200)),
        (f"Wall ticks: {a.wall_ticks}  UG: {a.underground_ticks}", (200, 200, 200)),
        (f"Dist carried: {a.dist_carried:.0f}", (200, 200, 200)),
        (f"Room: {room_name}  Loops: {a._loop_penalties}", (200, 200, 200)),
        (f"Stack: {len(a.stack)}/{MAX_CARRY}  {' '.join(TILE_NAMES.get(t, '?') for t in a.stack)}", (200, 200, 200)),
        (f"Climbing: {'yes' if a.on_wall else 'no'}  UG: {'yes' if a.underground else 'no'}", (200, 200, 200)),
    ]
    for text, col in stat_lines:
        sim.screen.blit(f.render(text, True, col), (x, y))
        y += lh
    y += 6
    pygame.draw.line(sim.screen, (100, 100, 140), (x, y), (x + panel_w - 20, y))
    y += 6
    sim.screen.blit(f.render("Neural Net Outputs:", True, (180, 180, 220)), (x, y))
    y += lh + 2
    labels = ["Move Left", "Move Right", "Climb Up", "Dig", "Pick Up", "Drop"]
    for i, label in enumerate(labels):
        val = float(a._last_nn_out[i])
        if val >= 0:
            bar_col, fill_w, bar_x = (80, 200, 80), int(val * 70), x + 220
        else:
            fill_w = int((-val) * 70)
            bar_col, bar_x = (200, 80, 80), x + 220 - fill_w
        sim.screen.blit(f.render(label, True, (180, 180, 180)), (x, y))
        pygame.draw.rect(sim.screen, (40, 40, 50), (x + 150, y + 1, 140, 12))
        pygame.draw.line(sim.screen, (100, 100, 100), (x + 220, y), (x + 220, y + 12))
        if fill_w > 0:
            pygame.draw.rect(sim.screen, bar_col, (bar_x, y + 1, fill_w, 12))
        sim.screen.blit(f.render(f"{val:+.2f}", True, bar_col), (x + 298, y))
        y += lh + 2


def draw_ui(sim):
    font = sim._font_ui
    now, sky = datetime.datetime.now(), sky_colour()
    bri = (sky[0] + sky[1] + sky[2]) / 3
    phase = "night" if bri < 40 else ("dusk/dawn" if bri < 120 else "day")
    carrying = sum(1 for a in sim.ants if a.stack)
    climbing = sum(1 for a in sim.ants if a.on_wall)
    ds = sim._last_death_summary
    death_str = (f"Last gen — border:{ds.get('border', 0)}  starved:{ds.get('starvation', 0)}  spider:{ds.get('spider', 0)}  cavein:{ds.get('cavein', 0)}  looping:{ds.get('looping', 0)}") if any(v for v in ds.values()) else ""
    stag_str = f"  [STAGNATION x{sim._stagnation_count} — mutation spiked]" if sim._stagnation_count >= 5 else ""
    avg_fit = f"{sim._fitness_history[-1]:.1f}" if sim._fitness_history else "—"
    lines = [
        f"Gen {sim.gen}  |  Ants: {len(sim.ants)}/{COLONY_SIZE}  (wipe if <{MIN_SURVIVORS})  |  Speed: {sim.speed}x",
        death_str,
        f"Avg fitness: {avg_fit}{stag_str}",
        f"Dug: {sim.stats['dug']}  Cave-ins: {sim.stats['caveins']}",
        f"Food eaten: {sim.stats['food_eaten']}  Delivered: {sim.stats['food_delivered']}",
        f"Aphids killed: {sim.stats['aphids_killed']}  Milked: {sim.stats['aphids_milked']}  Farmed: {sim.stats['aphids_farmed']}",
        f"Carrying: {carrying}  Climbing: {climbing}  Spider kills: {sim.stats['spider_kills']}",
        f"Tunnels near nest: {sim.stats['tunnels_near_nest']}",
        f"Rooms — nest:{sim.room_detector.counts.get(2, 0)}  farm:{sim.room_detector.counts.get(3, 0)}  chamber:{sim.room_detector.counts.get(5, 0)}",
        f"Dig loops: {sim.stats['dig_loops']}  Drop loops: {sim.stats['drop_loops']}",
        f"Picked up: {sim.stats['tiles_picked_up']}  Dropped: {sim.stats['tiles_dropped']}",
        f"Time: {now.strftime('%H:%M')}  Sky: {phase}",
        f"Save: {os.path.basename(sim.save_dir)}",
        "", "[1-5] Speed  [S] Save  [R] Reset  [Q] Quit", "[F] Fullscreen  [Scroll] Zoom  [Right-drag] Pan",
        f"Zoom: {sim.camera_zoom:.1f}x  |  Click ant = follow",
    ]
    txt_col = (255, 255, 200) if bri < 80 else (255, 255, 255)
    for i, line in enumerate(lines):
        sim.screen.blit(font.render(line, True, (0, 0, 0)), (21, 11 + i * 16))
        sim.screen.blit(font.render(line, True, txt_col), (20, 10 + i * 16))
