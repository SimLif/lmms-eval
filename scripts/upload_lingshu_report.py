"""Upload Lingshu paper benchmark results to a new Google Sheets tab.

Creates (or overwrites) a tab named 'Lingshu Report' in the same spreadsheet.

Usage:
    uv run python scripts/upload_lingshu_report.py
    uv run python scripts/upload_lingshu_report.py --dry-run
    uv run python scripts/upload_lingshu_report.py --target "_test_lingshu"
"""
from __future__ import annotations

import argparse
import os
from typing import Any

SPREADSHEET_ID = "1JK1Fev_Lc7ra8Z6OoSQjS07PMSJ1uJ5vudDYLGrTVPc"
CREDENTIALS_PATH = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS",
    ".secrets/google_credentials.json",
)

# Columns: Model, MMMU-Med, VQA-RAD, SLAKE, PathVQA, PMC-VQA, OmniMedVQA, MedXpertQA, Avg.
HEADERS = ["Model", "MMMU-Med", "VQA-RAD", "SLAKE", "PathVQA", "PMC-VQA", "OmniMedVQA", "MedXpertQA", "Avg."]
METRIC_COLS = list(range(1, 8))   # columns 1-7 (B-H) are numeric metrics
AVG_COL = 8                        # column 8 (I)

# Raw table data: (display_name, mmmu, vqa_rad, slake, pathvqa, pmc_vqa, omnimedvqa, medxpertqa, avg)
# None = missing value (shown as "-")
TABLE: list[tuple] = [
    # ── Section header ──────────────────────────────────────────────────────────
    ("Proprietary Models", None, None, None, None, None, None, None, None),
    ("GPT-4.1",           75.2, 65.0, 72.2, 55.5, 55.2, 75.5, 45.2, 63.4),
    ("Claude Sonnet 4",   74.6, 67.6, 70.6, 54.2, 54.4, 65.5, 43.3, 61.5),
    ("Gemini-2.5-Flash",  76.9, 68.5, 75.8, 55.4, 55.4, 71.0, 52.8, 65.1),
    # ── Section header ──────────────────────────────────────────────────────────
    ("Open-source Models (<10B)", None, None, None, None, None, None, None, None),
    ("BiomedGPT",         24.9, 16.6, 13.6, 11.3, 27.6, 27.9, None, None),
    ("Med-R1-2B",         34.8, 39.0, 54.5, 15.3, 47.4, None, 21.1, None),
    ("MedVLM-R1-2B",      35.2, 48.6, 56.0, 32.5, 47.6, 77.7, 20.4, 45.4),
    ("MedGemma-4B-IT",    43.7, 72.5, 76.4, 48.8, 49.9, 69.8, 22.3, 54.8),
    ("LLaVA-Med-7B",      29.3, 53.7, 48.0, 38.8, 30.5, 44.3, 20.3, 37.8),
    ("HuatuoGPT-V-7B",    47.3, 67.0, 67.8, 48.0, 53.3, 74.2, 21.6, 54.2),
    ("BioMediX2-8B",      39.8, 49.2, 57.7, 37.0, 43.5, 63.3, 21.8, 44.6),
    ("Qwen2.5VL-7B",      50.6, 64.5, 67.2, 44.1, 51.9, 63.6, 22.3, 52.0),
    ("InternVL2.5-8B",    53.5, 59.4, 69.0, 42.1, 51.3, 81.3, 21.7, 54.0),
    ("InternVL3-8B",      59.2, 65.4, 72.8, 48.6, 53.8, 79.1, 22.4, 57.3),
    ("Lingshu-7B",        54.0, 67.9, 83.1, 61.9, 56.3, 82.9, 26.7, 61.8),
    ("Lingshu-I-8B",      49.1, 73.0, 91.6, 74.9, 55.8, 79.7, 27.5, 64.5),
    # ── Section header ──────────────────────────────────────────────────────────
    ("Open-source Models (>10B)", None, None, None, None, None, None, None, None),
    ("HealthGPT-14B",     49.6, 65.0, 66.1, 56.7, 56.4, 75.2, 24.7, 56.2),
    ("HuatuoGPT-V-34B",   51.8, 61.4, 69.5, 44.4, 56.6, 74.0, 22.1, 54.3),
    ("MedDr-40B",         49.3, 65.2, 66.4, 53.5, 13.9, 64.3, None, None),
    ("InternVL3-14B",     63.1, 66.3, 72.8, 48.0, 54.1, 78.9, 23.1, 58.0),
    ("Qwen2.5V-32B",      59.6, 71.8, 71.2, 41.9, 54.5, 68.2, 25.2, 56.1),
    ("InternVL2.5-38B",   61.6, 61.4, 70.3, 46.9, 57.2, 79.9, 24.4, 57.4),
    ("InternVL3-38B",     65.2, 65.4, 72.7, 51.0, 56.6, 79.8, 25.2, 59.4),
    ("Lingshu-32B",       62.3, 76.5, 89.2, 65.9, 57.9, 83.4, 30.9, 66.6),
]

SECTION_HEADERS = {
    "Proprietary Models",
    "Open-source Models (<10B)",
    "Open-source Models (>10B)",
}

# Colors
HEADER_BG    = {"red": 0.216, "green": 0.255, "blue": 0.318}
HEADER_FG    = {"red": 1.0,   "green": 1.0,   "blue": 1.0}
SECTION_BG   = {"red": 0.95,  "green": 0.95,  "blue": 0.95}
LINGSHU_BG   = {"red": 1.000, "green": 0.969, "blue": 0.925}
RED_TEXT     = {"red": 0.86,  "green": 0.15,  "blue": 0.15}
WHITE        = {"red": 1.0,   "green": 1.0,   "blue": 1.0}


def get_sheets_service():
    import httplib2
    from google.oauth2 import service_account
    from google_auth_httplib2 import AuthorizedHttp
    from googleapiclient.discovery import build

    creds = service_account.Credentials.from_service_account_file(
        CREDENTIALS_PATH,
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    proxy_url = os.environ.get("https_proxy") or os.environ.get("http_proxy")
    if proxy_url:
        proxy_info = httplib2.proxy_info_from_url(proxy_url)
        http = httplib2.Http(proxy_info=proxy_info)
        authorized = AuthorizedHttp(creds, http=http)
        return build("sheets", "v4", http=authorized, cache_discovery=False)
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def get_or_create_sheet(service, spreadsheet_id: str, title: str) -> int:
    meta = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    for sheet in meta["sheets"]:
        if sheet["properties"]["title"] == title:
            return sheet["properties"]["sheetId"]
    resp = service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"requests": [{"addSheet": {"properties": {"title": title}}}]},
    ).execute()
    return resp["replies"][0]["addSheet"]["properties"]["sheetId"]


def cell_value(v: float | None) -> dict:
    if v is None:
        return {"userEnteredValue": {"stringValue": "-"}}
    return {"userEnteredValue": {"numberValue": v},
            "userEnteredFormat": {"numberFormat": {"type": "NUMBER", "pattern": "0.0"}}}


def build_rows() -> list[list[dict]]:
    rows = []
    # Header row
    header_cells = []
    for h in HEADERS:
        header_cells.append({
            "userEnteredValue": {"stringValue": h},
            "userEnteredFormat": {
                "backgroundColor": HEADER_BG,
                "textFormat": {"bold": True, "foregroundColor": HEADER_FG},
                "horizontalAlignment": "CENTER",
            },
        })
    rows.append(header_cells)

    for entry in TABLE:
        name = entry[0]
        vals = entry[1:]
        is_section = name in SECTION_HEADERS
        is_lingshu = name.startswith("Lingshu")

        if is_section:
            # Section header: span all columns, bold, grey background
            cells = [{
                "userEnteredValue": {"stringValue": name},
                "userEnteredFormat": {
                    "backgroundColor": SECTION_BG,
                    "textFormat": {"bold": True},
                    "horizontalAlignment": "LEFT",
                },
            }] + [{"userEnteredValue": {"stringValue": ""}, "userEnteredFormat": {"backgroundColor": SECTION_BG}}
                  for _ in range(8)]
            rows.append(cells)
            continue

        bg = LINGSHU_BG if is_lingshu else WHITE
        row = [{
            "userEnteredValue": {"stringValue": name},
            "userEnteredFormat": {
                "backgroundColor": bg,
                "textFormat": {"bold": is_lingshu},
            },
        }]
        for v in vals:
            c = cell_value(v)
            c.setdefault("userEnteredFormat", {})["backgroundColor"] = bg
            if is_lingshu:
                c["userEnteredFormat"].setdefault("textFormat", {})["bold"] = True
            row.append(c)
        rows.append(row)

    return rows


def find_column_maxes(data_rows: list[tuple]) -> dict[int, float]:
    """Find max value per metric column (cols 1-7)."""
    maxes: dict[int, float] = {}
    for entry in data_rows:
        if entry[0] in SECTION_HEADERS:
            continue
        for col_idx in METRIC_COLS + [AVG_COL]:
            v = entry[col_idx]
            if v is not None:
                if col_idx not in maxes or v > maxes[col_idx]:
                    maxes[col_idx] = v
    return maxes


def build_format_requests(sheet_id: int) -> list[dict]:
    requests = []

    # Freeze row 1
    requests.append({"updateSheetProperties": {
        "properties": {"sheetId": sheet_id,
                        "gridProperties": {"frozenRowCount": 1}},
        "fields": "gridProperties.frozenRowCount",
    }})

    # Column widths
    widths = {0: 200, **{c: 90 for c in range(1, 9)}}
    for col, w in widths.items():
        requests.append({"updateDimensionProperties": {
            "range": {"sheetId": sheet_id, "dimension": "COLUMNS",
                      "startIndex": col, "endIndex": col + 1},
            "properties": {"pixelSize": w},
            "fields": "pixelSize",
        }})

    # Gradient on metric columns (B-I)
    for col in range(1, 9):
        requests.append({"addConditionalFormatRule": {
            "rule": {
                "ranges": [{"sheetId": sheet_id, "startColumnIndex": col, "endColumnIndex": col + 1}],
                "gradientRule": {
                    "minpoint": {"type": "MIN", "color": WHITE},
                    "midpoint": {"type": "PERCENTILE", "value": "50",
                                 "color": {"red": 0.863, "green": 0.984, "blue": 0.906}},
                    "maxpoint": {"type": "MAX",
                                 "color": {"red": 0.082, "green": 0.635, "blue": 0.286}},
                },
            },
            "index": 0,
        }})

    # Bold red text for column max values
    maxes = find_column_maxes(TABLE)
    row_offset = 1  # skip header
    for i, entry in enumerate(TABLE):
        if entry[0] in SECTION_HEADERS:
            row_offset_adj = row_offset + i
            continue
        row_idx = row_offset + i
        for col_idx, max_val in maxes.items():
            v = entry[col_idx]
            if v is not None and v == max_val:
                requests.append({"repeatCell": {
                    "range": {"sheetId": sheet_id,
                              "startRowIndex": row_idx, "endRowIndex": row_idx + 1,
                              "startColumnIndex": col_idx, "endColumnIndex": col_idx + 1},
                    "cell": {"userEnteredFormat": {"textFormat": {"bold": True, "foregroundColor": RED_TEXT}}},
                    "fields": "userEnteredFormat.textFormat",
                }})

    return requests


def upload(target: str, dry_run: bool) -> None:
    rows = build_rows()
    if dry_run:
        print(f"[DRY-RUN] Would write {len(rows)} rows to tab '{target}'")
        for r in rows[:4]:
            vals = [c.get("userEnteredValue", {}).get("stringValue") or
                    c.get("userEnteredValue", {}).get("numberValue", "") for c in r]
            print(f"  {vals}")
        return

    service = get_sheets_service()
    sheet_id = get_or_create_sheet(service, SPREADSHEET_ID, target)
    print(f"Sheet '{target}' id={sheet_id}")

    # Clear existing content
    service.spreadsheets().values().clear(
        spreadsheetId=SPREADSHEET_ID,
        range=f"'{target}'",
    ).execute()

    # Write data
    service.spreadsheets().batchUpdate(
        spreadsheetId=SPREADSHEET_ID,
        body={"requests": [{"updateCells": {
            "rows": [{"values": row} for row in rows],
            "fields": "userEnteredValue,userEnteredFormat",
            "start": {"sheetId": sheet_id, "rowIndex": 0, "columnIndex": 0},
        }}]},
    ).execute()
    print(f"Written {len(rows)} rows")

    # Apply formatting
    fmt_reqs = build_format_requests(sheet_id)
    if fmt_reqs:
        service.spreadsheets().batchUpdate(
            spreadsheetId=SPREADSHEET_ID,
            body={"requests": fmt_reqs},
        ).execute()
    print("Formatting applied")
    print(f"Done: https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="Lingshu Report")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    upload(args.target, args.dry_run)


if __name__ == "__main__":
    main()
