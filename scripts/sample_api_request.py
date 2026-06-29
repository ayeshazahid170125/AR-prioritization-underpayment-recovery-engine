"""Small smoke-test client for the Project 02 FastAPI app.

Run after starting the API:
    python scripts/sample_api_request.py
"""

import json

import requests


API_URL = "http://localhost:8001/predict/recovery"

payload = {
    "procedure_category": "Evaluation_and_Management",
    "payer_type_proxy": "Medicare_Participating",
    "place_of_service": "O",
    "tot_srvcs": 75,
    "tot_benes": 42,
    "avg_sbmtd_chrg": 325.0,
    "review_flag": 1,
    "duplicate_key_flag": 0,
    "zero_expected_reason_flag": 0,
    "locality_count": 3.0,
    "ruca_unknown_flag": 0,
}


def main():
    response = requests.post(API_URL, json=payload, timeout=30)
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
