"""Generate a personalized daily forecast from a date of birth using ChatGPT.

Usage:
    export OPENAI_API_KEY="..."
    python dob_forecast_chatgpt.py --dob 1995-08-14
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

from openai import OpenAI


DATE_FMT = "%Y-%m-%d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Accept a date of birth and return today's forecast via ChatGPT."
    )
    parser.add_argument(
        "--dob",
        required=True,
        help="Date of birth in YYYY-MM-DD format (example: 1998-12-31).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model name to use (default: gpt-4o-mini).",
    )
    return parser.parse_args()


def parse_dob(dob_raw: str) -> dt.date:
    try:
        return dt.datetime.strptime(dob_raw, DATE_FMT).date()
    except ValueError as exc:
        raise ValueError(
            f"Invalid --dob value '{dob_raw}'. Use YYYY-MM-DD format."
        ) from exc


def build_prompt(dob: dt.date, today: dt.date) -> str:
    return (
        "You are a friendly daily forecast assistant. "
        f"The user's date of birth is {dob.isoformat()} and today's date is {today.isoformat()}. "
        "Give a short, positive 'today forecast' with these sections: "
        "Energy, Career/Study, Relationships, and One Action Step. "
        "Keep it practical, 120 words max, and avoid medical/financial/legal claims."
    )


def generate_forecast(dob: dt.date, model: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=api_key)
    today = dt.date.today()

    response = client.responses.create(
        model=model,
        input=build_prompt(dob=dob, today=today),
        temperature=0.8,
    )
    return response.output_text.strip()


def main() -> int:
    args = parse_args()

    try:
        dob = parse_dob(args.dob)
        forecast = generate_forecast(dob=dob, model=args.model)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("\n=== Your Forecast for Today ===")
    print(forecast)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
