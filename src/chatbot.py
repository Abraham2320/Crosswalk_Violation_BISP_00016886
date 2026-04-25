"""
Interactive terminal chatbot for the Crosswalk Violation system.
Launched via:  python run_system.py --chatbot
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import anthropic

DB_PATH = Path(__file__).resolve().parent.parent / "crosswalk_violations.db"

SYSTEM_PROMPT = (
    "You are a traffic safety analyst assistant for a crosswalk violation detection "
    "system in Tashkent, Uzbekistan. You have access to violation data and help traffic "
    "authorities understand patterns and make decisions. Be concise, factual, and actionable."
)


def _load_stats() -> dict:
    """Load summary stats from the SQLite DB. Returns safe defaults when DB is absent."""
    if not DB_PATH.exists():
        return {
            "total_violations": 0,
            "top_5_vehicles": [],
            "peak_hour": "N/A",
            "plate_detection_rate": "N/A",
            "note": "Database not found — run the detection system first.",
        }

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    total = cur.execute("SELECT COUNT(*) AS n FROM violations").fetchone()["n"]

    top5 = cur.execute(
        "SELECT vehicle_id, COUNT(*) AS cnt FROM violations "
        "GROUP BY vehicle_id ORDER BY cnt DESC LIMIT 5"
    ).fetchall()
    top5_list = [{"vehicle_id": r["vehicle_id"], "count": r["cnt"]} for r in top5]

    peak_row = cur.execute(
        "SELECT CAST(strftime('%H', timestamp) AS INTEGER) AS hr, COUNT(*) AS cnt "
        "FROM violations GROUP BY hr ORDER BY cnt DESC LIMIT 1"
    ).fetchone()
    peak_hour = f"{peak_row['hr']:02d}:00" if peak_row else "N/A"

    plate_row = cur.execute(
        "SELECT "
        "  SUM(CASE WHEN plate_number IS NOT NULL AND plate_number != '' THEN 1 ELSE 0 END) AS detected, "
        "  COUNT(*) AS total "
        "FROM violations"
    ).fetchone()
    if plate_row and plate_row["total"] > 0:
        rate = plate_row["detected"] / plate_row["total"] * 100
        plate_rate = f"{rate:.1f}%"
    else:
        plate_rate = "N/A"

    conn.close()
    return {
        "total_violations": total,
        "top_5_vehicles": top5_list,
        "peak_hour": peak_hour,
        "plate_detection_rate": plate_rate,
    }


def run_chatbot() -> None:
    print("\n" + "=" * 60)
    print("  Crosswalk Violation System — AI Chatbot")
    print("  Type 'quit' or 'exit' to leave.")
    print("=" * 60)

    stats = _load_stats()
    print(f"\nLoaded database stats:")
    print(f"  Total violations : {stats['total_violations']}")
    print(f"  Peak hour        : {stats['peak_hour']}")
    print(f"  Plate detect rate: {stats['plate_detection_rate']}")
    if stats["top_5_vehicles"]:
        top_str = ", ".join(
            f"Vehicle {v['vehicle_id']} ({v['count']}x)"
            for v in stats["top_5_vehicles"]
        )
        print(f"  Top offenders    : {top_str}")
    if "note" in stats:
        print(f"  Note             : {stats['note']}")
    print()

    context_block = (
        f"Current violation database summary:\n"
        f"- Total violations recorded: {stats['total_violations']}\n"
        f"- Top 5 offending vehicles: "
        + (
            ", ".join(
                f"Vehicle {v['vehicle_id']} ({v['count']} violations)"
                for v in stats["top_5_vehicles"]
            ) or "N/A"
        )
        + f"\n- Peak violation hour: {stats['peak_hour']}\n"
        f"- License plate recognition rate: {stats['plate_detection_rate']}\n"
    )

    try:
        client = anthropic.Anthropic()
    except Exception as exc:
        print(f"Failed to initialise Anthropic client: {exc}")
        print("Make sure ANTHROPIC_API_KEY is set in your environment.")
        return

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye.")
            break

        full_user_content = f"{context_block}\nQuestion: {user_input}"

        try:
            response = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=800,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": full_user_content}],
            )
            answer = response.content[0].text
            print(f"\nAssistant: {answer}\n")
        except anthropic.AuthenticationError:
            print("Error: ANTHROPIC_API_KEY is missing or invalid.\n")
        except Exception as exc:
            print(f"API error: {exc}\n")
