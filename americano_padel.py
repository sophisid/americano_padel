import csv
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class Match:
    match_id: int
    team1_p1: str
    team1_p2: str
    team2_p1: str
    team2_p2: str



def round_robin_pairings(n: int) -> List[List[Tuple[int, int]]]:
    """
    Generate round-robin pairings on n players (n even), using the 'circle method'.

    This produces n-1 rounds; in each round we get n/2 disjoint pairs (i.e., a perfect matching).
    Each unordered pair of players appears exactly once across all rounds.
    """
    players = list(range(n))
    rounds: List[List[Tuple[int, int]]] = []

    for _ in range(n - 1):
        pairs: List[Tuple[int, int]] = []
        for i in range(n // 2):
            a = players[i]
            b = players[-(i + 1)]
            pairs.append((a, b))
        rounds.append(pairs)

        # rotate players except first
        players = [players[0]] + [players[-1]] + players[1:-1]

    return rounds


def generate_teammate_once_schedule(players: List[str]) -> List[Match]:
    """
    STRICT mode:
    Generate a schedule where each pair of players are teammates exactly once.
    Requires len(players) to be a multiple of 4.
    """
    n = len(players)
    if n < 4 or n % 2 != 0:
        raise ValueError("Number of players must be even and at least 4.")

    if n % 4 != 0:
        raise ValueError(
            "Strict mode: number of players must be a multiple of 4 "
            "(for perfect 'each pair teammates exactly once'). "
            "Use --loose for other even numbers."
        )

    idx_rounds = round_robin_pairings(n)  # list of rounds, each with n/2 pairs

    matches: List[Match] = []
    match_id = 1

    # For each round, we have n/2 pairs (teammates).
    # Group these pairs into matches of 4 players: (pair0 vs pair1), (pair2 vs pair3), ...
    for rnd in idx_rounds:
        if len(rnd) % 2 != 0:
            raise RuntimeError("Unexpected: round has odd number of pairs.")

        for i in range(0, len(rnd), 2):
            (a, b) = rnd[i]
            (c, d) = rnd[i + 1]

            m = Match(
                match_id=match_id,
                team1_p1=players[a],
                team1_p2=players[b],
                team2_p1=players[c],
                team2_p2=players[d],
            )
            matches.append(m)
            match_id += 1

    return matches


def generate_fully_balanced_schedule_4(players: List[str]) -> List[Match]:
    """
    Special case: 4 players.
    We can satisfy:
      - each pair are teammates exactly once
      - each pair are opponents exactly twice
    This is the classic 3-match Americano mini format.
    """
    if len(players) != 4:
        raise ValueError("Fully balanced opponents constraint is only implemented for exactly 4 players.")

    A, B, C, D = players

    matches = [
        Match(1, A, B, C, D),
        Match(2, A, C, B, D),
        Match(3, A, D, B, C),
    ]
    return matches


def generate_balanced_schedule(players: List[str]) -> List[Match]:
    """
    BALANCED mode for N % 4 == 0.

    Goal:
      - For any N (multiple of 4):
          each unordered pair of players is teammates exactly (N / 4) times.
      - For N = 4: this also implies each pair are opponents exactly 2 times (perfect Americano).
      - For N > 4: opponents are more evenly distributed than in strict mode,
        but we do NOT guarantee "exactly K times" for all pairs of opponents.
        That becomes a hard combinatorial design problem.

    Construction:
      - Build a base STRICT schedule where each pair is teammates exactly once.
      - Let multiplier = N / 4.
      - Repeat the base schedule 'multiplier' times, each time with a cyclic rotation
        of player indices. This keeps teammate frequencies uniform and increases
        variety of opponents.
    """
    n = len(players)
    if n < 4 or n % 4 != 0:
        raise ValueError("Balanced mode: number of players must be a multiple of 4 and at least 4.")

    # N = 4: we already have a perfect closed-form schedule
    if n == 4:
        return generate_fully_balanced_schedule_4(players)

    # Base schedule on indices (strict teammate-once schedule)
    idx_rounds = round_robin_pairings(n)
    base_matches: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    for rnd in idx_rounds:
        for i in range(0, len(rnd), 2):
            (a, b) = rnd[i]
            (c, d) = rnd[i + 1]
            base_matches.append(((a, b), (c, d)))

    multiplier = n // 4

    matches: List[Match] = []
    match_id = 1

    # Repeat with cyclic rotations
    for k in range(multiplier):
        offset = k  # could be any step coprime with n; 1 is fine
        def f(x: int) -> int:
            return (x + offset) % n

        for (a, b), (c, d) in base_matches:
            m = Match(
                match_id=match_id,
                team1_p1=players[f(a)],
                team1_p2=players[f(b)],
                team2_p1=players[f(c)],
                team2_p2=players[f(d)],
            )
            matches.append(m)
            match_id += 1

    return matches


def generate_loose_schedule(players: List[str]) -> List[Match]:
    """
    LOOSE mode:
    Accept any even number of players >= 4.
    Simple symmetric heuristic:
      - Arrange players in a circle.
      - For each r in [0, n-1], take 4 consecutive players:
            [r, r+1, r+2, r+3] mod n
        and define a match:
            (p[r], p[r+1]) vs (p[r+2], p[r+3])
    Each player appears in several matches with different teammates/opponents.
    We do NOT guarantee 'each pair teammates exactly once'.
    """
    n = len(players)
    if n < 4 or n % 2 != 0:
        raise ValueError("Loose mode: number of players must be even and at least 4.")

    matches: List[Match] = []
    match_id = 1

    for r in range(n):
        a = players[r % n]
        b = players[(r + 1) % n]
        c = players[(r + 2) % n]
        d = players[(r + 3) % n]

        m = Match(
            match_id=match_id,
            team1_p1=a,
            team1_p2=b,
            team2_p1=c,
            team2_p2=d,
        )
        matches.append(m)
        match_id += 1

    return matches


# ---------- CSV I/O ----------

def read_players_csv(path: str) -> List[str]:
    players: List[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "name" not in reader.fieldnames:
            raise ValueError("Players CSV must have a 'name' column.")
        for row in reader:
            name = row["name"].strip()
            if name:
                players.append(name)
    if len(players) < 4:
        raise ValueError("Need at least 4 players.")
    return players


def write_schedule_csv(matches: List[Match], path: str) -> None:
    fieldnames = [
        "match_id",
        "team1_p1",
        "team1_p2",
        "team2_p1",
        "team2_p2",
        "team1_score",
        "team2_score",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in matches:
            writer.writerow(
                {
                    "match_id": m.match_id,
                    "team1_p1": m.team1_p1,
                    "team1_p2": m.team1_p2,
                    "team2_p1": m.team2_p1,
                    "team2_p2": m.team2_p2,
                    "team1_score": "",  # to be filled later
                    "team2_score": "",
                }
            )


def compute_scores_from_schedule_csv(path: str) -> Dict[str, int]:
    """
    Reads the schedule+scores CSV and computes per-player total points.

    For each match:
      - Every player gets the number of points scored by their own team.
      - If scores are empty / missing, the match is ignored for scoring.
    """
    totals: Dict[str, int] = {}

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {
            "team1_p1",
            "team1_p2",
            "team2_p1",
            "team2_p2",
            "team1_score",
            "team2_score",
        }
        if not required_cols.issubset(reader.fieldnames or []):
            missing = required_cols - set(reader.fieldnames or [])
            raise ValueError(f"Scores CSV is missing required columns: {missing}")

        for row in reader:
            t1p1 = row["team1_p1"].strip()
            t1p2 = row["team1_p2"].strip()
            t2p1 = row["team2_p1"].strip()
            t2p2 = row["team2_p2"].strip()

            s1_raw = row["team1_score"].strip()
            s2_raw = row["team2_score"].strip()

            # Skip matches with no scores
            if not s1_raw or not s2_raw:
                continue

            try:
                s1 = int(s1_raw)
                s2 = int(s2_raw)
            except ValueError:
                raise ValueError(f"Invalid score values: '{s1_raw}', '{s2_raw}' in row {row}")

            for player in [t1p1, t1p2, t2p1, t2p2]:
                if player not in totals:
                    totals[player] = 0

            # Americano style: each player gets their team's points
            totals[t1p1] += s1
            totals[t1p2] += s1
            totals[t2p1] += s2
            totals[t2p2] += s2

    return totals


# ---------- CLI ----------

def cmd_generate(args: argparse.Namespace) -> None:
    players = read_players_csv(args.players_csv)

    if args.loose:
        if args.balanced:
            print(
                "Warning: --balanced is ignored in --loose mode.\n"
                "Loose mode uses a heuristic rotation and does not enforce teammate counts."
            )
        matches = generate_loose_schedule(players)

    else:
        # STRICT / BALANCED mode
        if args.balanced:
            matches = generate_balanced_schedule(players)
        else:
            matches = generate_teammate_once_schedule(players)

    write_schedule_csv(matches, args.output_csv)
    print(f"Schedule written to {args.output_csv}")
    print(f"Total matches: {len(matches)}")


def cmd_scores(args: argparse.Namespace) -> None:
    totals = compute_scores_from_schedule_csv(args.schedule_csv)
    # Sort by descending total points
    sorted_totals = sorted(totals.items(), key=lambda x: x[1], reverse=True)

    print("Player rankings (total points):")
    for i, (player, points) in enumerate(sorted_totals, start=1):
        print(f"{i:2d}. {player}: {points} points")


def main():
    parser = argparse.ArgumentParser(description="Americano padel scheduler and scorer.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate
    p_gen = subparsers.add_parser("generate", help="Generate schedule from players CSV.")
    p_gen.add_argument("players_csv", help="Path to players CSV (with 'name' column).")
    p_gen.add_argument(
        "-o",
        "--output-csv",
        default="schedule.csv",
        help="Output CSV for schedule (default: schedule.csv).",
    )
    p_gen.add_argument(
        "--balanced",
        action="store_true",
        help=(
            "Balanced mode (only for N multiple of 4): "
            "for N=4, perfect Americano (teammates once, opponents twice). "
            "For N>4, each pair of players is teammates exactly N/4 times; "
            "opponents are more evenly distributed but not perfectly equal."
        ),
    )
    p_gen.add_argument(
        "--loose",
        action="store_true",
        help=(
            "Loose mode: allow any even number of players >= 4. "
            "Uses a simple symmetric rotation heuristic instead of perfect 'teammates-once' design."
        ),
    )
    p_gen.set_defaults(func=cmd_generate)

    # scores
    p_scores = subparsers.add_parser("scores", help="Compute rankings from schedule+scores CSV.")
    p_scores.add_argument("schedule_csv", help="Path to schedule CSV with filled-in scores.")
    p_scores.set_defaults(func=cmd_scores)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
