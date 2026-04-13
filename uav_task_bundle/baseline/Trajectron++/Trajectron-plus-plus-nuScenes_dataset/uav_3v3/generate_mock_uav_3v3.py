import argparse
import csv
from pathlib import Path

import numpy as np


TEAM_SPECS = (
    ("BLUE", 1.0, -120.0, -40.0, 18.0),
    ("RED", -1.0, 120.0, 40.0, -18.0),
)


def build_episode_rows(episode_id, timesteps, kill_plan):
    rows = []
    phase = episode_id * 0.25
    altitude_bias = 30.0 + 4.0 * episode_id

    for team_idx, (team, direction, base_x, base_y, base_vx) in enumerate(TEAM_SPECS):
        for slot in range(3):
            drone_id = f"{team.lower()}_{slot}"
            kill_timestep = kill_plan.get(drone_id)
            start_y = base_y + slot * 30.0
            start_z = altitude_bias + slot * 6.0 + team_idx * 8.0

            for timestep in range(timesteps):
                if kill_timestep is not None and timestep > kill_timestep:
                    break

                progress = float(timestep)
                x = base_x + base_vx * progress + direction * 5.0 * np.sin(0.12 * progress + phase + slot)
                y = start_y + 7.0 * np.sin(0.18 * progress + slot * 0.7 + phase)
                z = start_z + 3.0 * np.sin(0.09 * progress + team_idx + slot * 0.4)

                vx = base_vx + direction * 0.6 * np.cos(0.12 * progress + phase + slot)
                vy = 1.26 * np.cos(0.18 * progress + slot * 0.7 + phase)
                vz = 0.27 * np.cos(0.09 * progress + team_idx + slot * 0.4)

                hp = 100.0
                if kill_timestep is not None:
                    hp = max(0.0, 100.0 - (100.0 / max(kill_timestep, 1)) * progress)

                rows.append(
                    {
                        "episode_id": f"episode_{episode_id:03d}",
                        "timestep": timestep,
                        "drone_id": drone_id,
                        "team": team,
                        "x": round(x, 6),
                        "y": round(y, 6),
                        "z": round(z, 6),
                        "vx": round(vx, 6),
                        "vy": round(vy, 6),
                        "vz": round(vz, 6),
                        "hp": round(hp, 6),
                    }
                )

    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate mock 3v3 UAV trajectory data.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "mock_uav_3v3.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--num-episodes", type=int, default=12, help="Number of mock episodes to generate.")
    parser.add_argument("--timesteps", type=int, default=18, help="Timesteps per episode before kill truncation.")
    args = parser.parse_args()

    kill_templates = (
        {},
        {"blue_2": 9},
        {"red_1": 11},
        {"blue_0": 8, "red_2": 13},
    )

    rows = []
    for episode_id in range(args.num_episodes):
        kill_plan = kill_templates[episode_id % len(kill_templates)]
        rows.extend(build_episode_rows(episode_id, args.timesteps, kill_plan))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output_csv}")
    print(f"Episodes: {args.num_episodes}, drones per episode: 6")


if __name__ == "__main__":
    main()
