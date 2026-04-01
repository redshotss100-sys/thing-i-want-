import datetime
import json
import os
import pickle
import traceback
from collections import defaultdict

import numpy as np

from brain import brain_stats
from config import APHID_MAX_MILK_STOCK, SAVES_ROOT, TILE_NAMES
from entities import Aphid, Spider


def write_crash_dump(exc_info, extra=""):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SAVES_ROOT, f"crash_{ts}.txt")
    os.makedirs(SAVES_ROOT, exist_ok=True)
    with open(path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"ANT SIM CRASH DUMP  {datetime.datetime.now()}\n")
        f.write("=" * 60 + "\n\n")
        if extra:
            f.write(extra + "\n\n")
        f.write(traceback.format_exc())
    print(f"[CRASH] Dump written: {path}")
    return path


class StatsLogger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.log_path = os.path.join(save_dir, "stats_log.jsonl")
        self.all_gens = []

    def log_generation(self, gen, ants, sim_stats, exploit_flags):
        if not ants:
            return {}
        ages = [a.age for a in ants]
        fits = [a.fitness for a in ants]
        dug_l = [a.tiles_dug for a in ants]
        wall_t = [a.wall_ticks for a in ants]
        ug_t = [a.underground_ticks for a in ants]
        dist_c = [a.dist_carried for a in ants]
        carry_totals = defaultdict(int)
        for a in ants:
            for tile, cnt in a.tiles_carried.items():
                carry_totals[TILE_NAMES.get(tile, str(tile))] += cnt
        sorted_ants = sorted(ants, key=lambda a: a.fitness, reverse=True)
        top_brains = [brain_stats(a.weights) for a in sorted_ants[:3]]
        aphid_total = sim_stats['aphids_killed'] + sim_stats['aphids_milked']
        record = {
            "gen": gen,
            "timestamp": datetime.datetime.now().isoformat(),
            "colony_size": len(ants),
            "avg_lifespan": float(np.mean(ages)),
            "max_lifespan": int(max(ages)),
            "min_lifespan": int(min(ages)),
            "avg_fitness": float(np.mean(fits)),
            "max_fitness": float(max(fits)),
            "min_fitness": float(min(fits)),
            "tiles_dug_this_gen": int(np.sum(dug_l)),
            "avg_dug_per_ant": float(np.mean(dug_l)),
            "food_eaten": sim_stats['food_eaten'],
            "food_delivered": sim_stats.get('food_delivered', 0),
            "aphids_killed": sim_stats['aphids_killed'],
            "aphids_milked": sim_stats['aphids_milked'],
            "aphid_kill_ratio": round(sim_stats['aphids_killed'] / max(1, aphid_total), 3),
            "aphids_farmed": sim_stats.get('aphids_farmed', 0),
            "tunnels_near_nest": sim_stats.get('tunnels_near_nest', 0),
            "caveins": sim_stats['caveins'],
            "spider_kills": sim_stats['spider_kills'],
            "tiles_picked_up": sim_stats['tiles_picked_up'],
            "tiles_dropped": sim_stats['tiles_dropped'],
            "avg_wall_ticks": float(np.mean(wall_t)),
            "avg_underground_ticks": float(np.mean(ug_t)),
            "avg_dist_carried": float(np.mean(dist_c)),
            "carry_breakdown": dict(carry_totals),
            "top3_brain_stats": top_brains,
            "exploit_flags": exploit_flags,
            "deaths_by_cause": {
                "border": sim_stats.get('deaths_border', 0),
                "starvation": sim_stats.get('deaths_starvation', 0),
                "spider": sim_stats.get('deaths_spider', 0),
                "cavein": sim_stats.get('deaths_cavein', 0),
            },
            "dig_loops": sim_stats.get('dig_loops', 0),
            "drop_loops": sim_stats.get('drop_loops', 0),
            "rooms_nest": sim_stats.get('rooms_nest', 0),
            "rooms_farm": sim_stats.get('rooms_farm', 0),
            "rooms_chamber": sim_stats.get('rooms_chamber', 0),
        }
        self.all_gens.append(record)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def write_summary(self):
        if not self.all_gens:
            return
        gens = self.all_gens
        summary = {
            "total_generations": len(gens),
            "created": gens[0]["timestamp"],
            "last_updated": gens[-1]["timestamp"],
            "best_gen_by_fitness": max(gens, key=lambda g: g["max_fitness"])["gen"],
            "best_gen_by_lifespan": max(gens, key=lambda g: g["avg_lifespan"])["gen"],
            "total_tiles_dug": sum(g["tiles_dug_this_gen"] for g in gens),
            "total_food_eaten": sum(g["food_eaten"] for g in gens),
            "total_aphids_killed": sum(g["aphids_killed"] for g in gens),
            "total_aphids_milked": sum(g["aphids_milked"] for g in gens),
            "total_caveins": sum(g["caveins"] for g in gens),
            "total_spider_kills": sum(g["spider_kills"] for g in gens),
            "total_food_delivered": sum(g.get("food_delivered", 0) for g in gens),
            "total_aphids_farmed": sum(g.get("aphids_farmed", 0) for g in gens),
            "total_tunnels_near_nest": sum(g.get("tunnels_near_nest", 0) for g in gens),
            "total_dig_loops": sum(g.get("dig_loops", 0) for g in gens),
            "total_drop_loops": sum(g.get("drop_loops", 0) for g in gens),
            "fitness_over_time": [g["avg_fitness"] for g in gens],
            "lifespan_over_time": [g["avg_lifespan"] for g in gens],
            "dug_over_time": [g["tiles_dug_this_gen"] for g in gens],
            "exploit_flags_over_time": [g.get("exploit_flags", []) for g in gens],
        }
        with open(os.path.join(self.save_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)


def new_save_dir():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(SAVES_ROOT, f"run_{ts}")
    os.makedirs(d, exist_ok=True)
    return d


def load_simulation(screen, save_dir):
    from simulation import Simulation

    with open(os.path.join(save_dir, "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(save_dir, "brains.pkl"), "rb") as f:
        brains = pickle.load(f)

    sim = Simulation(screen, save_dir, gene_pool=brains["gene_pool"], start_gen=brains["gen"])
    sim.best_weights = brains.get("best_weights", [])
    sim._pool_snapshots = brains.get("snapshots", [])
    sim.speed = meta.get("speed", 1)
    sim._build_world()
    sim.surface_y = meta["surface_y"]
    sim.nest_x, sim.nest_y = meta["nest_x"], meta["nest_y"]
    for fname, attr in [("phero_food.npy", "food"), ("phero_danger.npy", "danger"), ("phero_home.npy", "home")]:
        fp = os.path.join(save_dir, fname)
        if os.path.exists(fp):
            setattr(sim.phero, attr, np.load(fp))
    rg = os.path.join(save_dir, "room_grid.npy")
    if os.path.exists(rg):
        sim.room_detector.grid = np.load(rg)
    th = os.path.join(save_dir, "tunnel_history.npy")
    if os.path.exists(th):
        sim.tunnel_history = np.load(th)

    sim.aphids = []
    for rec in meta.get("aphids", []):
        if len(rec) == 4:
            ax, ay, mc, ms = rec
        else:
            ax, ay, mc = rec
            ms = APHID_MAX_MILK_STOCK
        a = Aphid(ax, ay)
        a.milk_cooldown = mc
        a.milk_stock = ms
        sim.aphids.append(a)

    sim.spiders = []
    for sx, sy2, svx, svy in meta.get("spiders", []):
        sp = Spider(sx, sy2)
        sp.vx = svx
        sp.vy = svy
        sim.spiders.append(sp)

    sim.stats = meta.get("stats", sim._blank_stats())
    lp = os.path.join(save_dir, "stats_log.jsonl")
    if os.path.exists(lp):
        with open(lp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sim.logger.all_gens.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return sim
