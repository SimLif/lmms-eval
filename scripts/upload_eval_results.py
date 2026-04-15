"""Upload med-eval results from local YAML to Google Sheets.

Strategy: Clear content + reset format → Write data → Apply format.
Preserves sheet history (no delete/recreate).

Usage:
    uv run python scripts/upload_eval_results.py
    uv run python scripts/upload_eval_results.py --target _test_upload
    uv run python scripts/upload_eval_results.py --dry-run
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import yaml

DEFAULT_DATA = "data/eval_results/med_eval_results.yaml"
CREDENTIALS_PATH = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS",
    ".secrets/google_credentials.json",
)

# Metric field keys (in YAML) → column indices start at 2
METRIC_KEYS = [
    "mmmu_med", "vqa_rad", "slake", "pathvqa",
    "pmc_vqa", "vqa_med", "omnimedvqa", "medxpertqa", "pubmedqa",
]

# Column layout: A=Model, B=Params, C-K=metrics, L=Avg, M=wandb_id, N=Description
COL_WIDTHS = {
    0: 200,   # Model
    1: 80,    # Params
    # 2-10: metrics (90 each)
    11: 90,   # Avg
    12: 120,  # wandb_id
    13: 335,  # Description
}
METRIC_COL_WIDTH = 90

# Model row background colors by group
GROUP_MODEL_COLORS = {
    "General Baselines": None,  # white (default)
    "Medical Specialist Baselines": {"red": 1.000, "green": 0.969, "blue": 0.925},
    "FGMoE Checkpoints": {"red": 0.933, "green": 0.965, "blue": 1.000},
}

# Special model background (overrides group color)
SPECIAL_MODEL_COLORS = {
    "Qwen3.5-2B-Think": {"red": 0.953, "green": 0.937, "blue": 0.984},
}

# Red text color for column max values
RED_TEXT = {"red": 0.86, "green": 0.15, "blue": 0.15}


def load_data(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_sheets_service():
    """Create Google Sheets API service with proxy support."""
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
    else:
        http = httplib2.Http()
    return build("sheets", "v4", http=AuthorizedHttp(creds, http=http))


# ---------------------------------------------------------------------------
# Data building — numbers stay as numbers, not strings
# ---------------------------------------------------------------------------


def build_summary_rows(data: dict) -> list[list[Any]]:
    """Build 2D array. Numeric values stay as float (not string)."""
    header = ["Model", "Params"] + data["metrics"] + ["Avg", "wandb_id", "Description"]
    rows: list[list[Any]] = [header]
    for group in data["summary"]["groups"]:
        rows.append([group["name"]])
        for m in group["models"]:
            row: list[Any] = [m["name"], str(m["params"])]
            for key in METRIC_KEYS:
                val = m.get(key, "")
                row.append(float(val) if isinstance(val, (int, float)) else val)
            avg = m.get("avg", "")
            row.append(float(avg) if isinstance(avg, (int, float)) else "")
            row.append(m.get("wandb_id", ""))
            row.append(m.get("description", ""))
            rows.append(row)
        rows.append([])
    return rows


def find_group_header_rows(rows: list[list]) -> list[int]:
    return [i for i, row in enumerate(rows) if i > 0 and len(row) == 1 and row[0]]


def find_data_rows(rows: list[list]) -> list[int]:
    """Find 0-based row indices that contain model data (have numeric values)."""
    return [
        i for i, row in enumerate(rows)
        if i > 0 and len(row) > 2 and isinstance(row[2], (int, float))
    ]


def find_column_max_rows(
    rows: list[list], data: dict,
) -> list[tuple[int, int]]:
    """Find (row, col) pairs for each group's column-max values.

    Returns cells that should be marked with red bold text.
    """
    red_cells = []
    metric_cols = list(range(2, 2 + len(METRIC_KEYS)))  # C through K

    # Process each group separately
    group_start = None
    group_rows: list[int] = []
    for i, row in enumerate(rows):
        if i == 0:
            continue
        is_group_header = len(row) == 1 and row[0]
        is_empty = len(row) == 0 or (len(row) == 1 and not row[0])
        is_data = len(row) > 2 and isinstance(row[2], (int, float))

        if is_group_header:
            # Process previous group
            if group_rows:
                _mark_group_maxes(rows, group_rows, metric_cols, red_cells)
            group_rows = []
        elif is_data:
            group_rows.append(i)
        elif is_empty and group_rows:
            _mark_group_maxes(rows, group_rows, metric_cols, red_cells)
            group_rows = []

    # Last group
    if group_rows:
        _mark_group_maxes(rows, group_rows, metric_cols, red_cells)

    return red_cells


def _mark_group_maxes(
    rows: list[list],
    group_row_indices: list[int],
    metric_cols: list[int],
    red_cells: list[tuple[int, int]],
) -> None:
    """For each metric column, find the max value row within the group."""
    for col in metric_cols:
        max_val = -float("inf")
        max_row = -1
        for ri in group_row_indices:
            val = rows[ri][col] if col < len(rows[ri]) else None
            if isinstance(val, (int, float)) and val > max_val:
                max_val = val
                max_row = ri
        if max_row >= 0:
            red_cells.append((max_row, col))


# ---------------------------------------------------------------------------
# Sheet operations
# ---------------------------------------------------------------------------


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


def clear_sheet(service, spreadsheet_id: str, sheet_id: int, title: str) -> None:
    """Clear content, cell formatting, and conditional format rules."""
    # 1. Clear values
    service.spreadsheets().values().clear(
        spreadsheetId=spreadsheet_id,
        range=f"{title}!A1:ZZ10000",
    ).execute()

    # 2. Reset cell formatting (do NOT set numberFormat to TEXT — that blocks gradients)
    service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"requests": [{
            "repeatCell": {
                "range": {"sheetId": sheet_id},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": {"red": 1, "green": 1, "blue": 1},
                    "textFormat": {
                        "bold": False, "italic": False,
                        "foregroundColor": {"red": 0, "green": 0, "blue": 0},
                    },
                    "horizontalAlignment": "LEFT",
                }},
                "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)",
            }
        }]},
    ).execute()

    # 3. Delete conditional format rules
    meta = service.spreadsheets().get(
        spreadsheetId=spreadsheet_id,
        fields="sheets.conditionalFormats,sheets.properties.sheetId",
    ).execute()
    for sheet in meta.get("sheets", []):
        if sheet["properties"]["sheetId"] != sheet_id:
            continue
        rules = sheet.get("conditionalFormats", [])
        if rules:
            requests = [
                {"deleteConditionalFormatRule": {"sheetId": sheet_id, "index": i}}
                for i in range(len(rules) - 1, -1, -1)
            ]
            service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body={"requests": requests},
            ).execute()
            print(f"  Cleared {len(rules)} conditional format rules")


def build_format_requests(
    sheet_id: int,
    rows: list[list],
    fmt: dict,
    data: dict,
) -> list[dict]:
    """Build all batchUpdate requests for formatting."""
    n_rows = len(rows)
    n_cols = len(rows[0])
    requests = []

    # --- Frozen rows/columns ---
    requests.append({
        "updateSheetProperties": {
            "properties": {
                "sheetId": sheet_id,
                "gridProperties": {
                    "frozenRowCount": fmt.get("frozen_rows", 1),
                    "frozenColumnCount": fmt.get("frozen_columns", 2),
                },
            },
            "fields": "gridProperties.frozenRowCount,gridProperties.frozenColumnCount",
        }
    })

    # --- Column widths ---
    for col_idx in range(n_cols):
        width = COL_WIDTHS.get(col_idx, METRIC_COL_WIDTH)
        requests.append({
            "updateDimensionProperties": {
                "range": {
                    "sheetId": sheet_id,
                    "dimension": "COLUMNS",
                    "startIndex": col_idx,
                    "endIndex": col_idx + 1,
                },
                "properties": {"pixelSize": width},
                "fields": "pixelSize",
            }
        })

    # --- Header row ---
    hdr = fmt.get("header", {})
    requests.append({
        "repeatCell": {
            "range": {
                "sheetId": sheet_id,
                "startRowIndex": 0, "endRowIndex": 1,
                "startColumnIndex": 0, "endColumnIndex": n_cols,
            },
            "cell": {"userEnteredFormat": {
                "backgroundColor": hdr.get("background", {}),
                "textFormat": {
                    "bold": True,
                    "foregroundColor": hdr.get("text_format", {}).get("foreground_color", {}),
                },
                "horizontalAlignment": "CENTER",
            }},
            "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)",
        }
    })

    # --- Group header rows ---
    grp = fmt.get("group_header", {})
    for row_idx in find_group_header_rows(rows):
        requests.append({
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": row_idx, "endRowIndex": row_idx + 1,
                    "startColumnIndex": 0, "endColumnIndex": n_cols,
                },
                "cell": {"userEnteredFormat": {
                    "textFormat": {"bold": True},
                }},
                "fields": "userEnteredFormat.textFormat.bold",
            }
        })

    # --- Data cells: center + number format ---
    requests.append({
        "repeatCell": {
            "range": {
                "sheetId": sheet_id,
                "startRowIndex": 1, "endRowIndex": n_rows,
                "startColumnIndex": 2, "endColumnIndex": n_cols - 2,
            },
            "cell": {"userEnteredFormat": {
                "horizontalAlignment": "CENTER",
                "numberFormat": {"type": "NUMBER", "pattern": "0.00"},
            }},
            "fields": "userEnteredFormat(horizontalAlignment,numberFormat)",
        }
    })

    # --- Model column: group-specific background colors ---
    current_group = None
    for i, row in enumerate(rows):
        if i == 0:
            continue
        if len(row) == 1 and row[0]:
            current_group = row[0]
            continue
        if len(row) > 2 and isinstance(row[2], (int, float)):
            model_name = row[0]
            # Check special model override first
            bg = SPECIAL_MODEL_COLORS.get(model_name)
            if bg is None and current_group:
                bg = GROUP_MODEL_COLORS.get(current_group)
            if bg:
                requests.append({
                    "repeatCell": {
                        "range": {
                            "sheetId": sheet_id,
                            "startRowIndex": i, "endRowIndex": i + 1,
                            "startColumnIndex": 0, "endColumnIndex": 1,
                        },
                        "cell": {"userEnteredFormat": {"backgroundColor": bg}},
                        "fields": "userEnteredFormat.backgroundColor",
                    }
                })

    # --- PubMedQA column: fixed grey background ---
    pubmedqa = fmt.get("pubmedqa_column", {})
    if pubmedqa:
        requests.append({
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 1, "endRowIndex": n_rows,
                    "startColumnIndex": 10, "endColumnIndex": 11,
                },
                "cell": {"userEnteredFormat": {
                    "backgroundColor": pubmedqa.get("background", {}),
                }},
                "fields": "userEnteredFormat.backgroundColor",
            }
        })

    # --- Gradient conditional formatting (per metric column) ---
    grad = fmt.get("gradient_rule", {})
    for col_idx in grad.get("columns_with_gradient", []):
        minp = grad["minpoint"]
        midp = grad["midpoint"]
        maxp = grad["maxpoint"]
        requests.append({
            "addConditionalFormatRule": {
                "rule": {
                    "ranges": [{
                        "sheetId": sheet_id,
                        "startRowIndex": 1, "endRowIndex": n_rows,
                        "startColumnIndex": col_idx, "endColumnIndex": col_idx + 1,
                    }],
                    "gradientRule": {
                        "minpoint": {"type": minp["type"], "color": minp["color"]},
                        "midpoint": {"type": midp["type"], "value": midp["value"], "color": midp["color"]},
                        "maxpoint": {"type": maxp["type"], "color": maxp["color"]},
                    },
                },
                "index": 0,
            }
        })

    # --- Red bold text for column-max values (per group) ---
    red_cells = find_column_max_rows(rows, data)
    for row_idx, col_idx in red_cells:
        requests.append({
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": row_idx, "endRowIndex": row_idx + 1,
                    "startColumnIndex": col_idx, "endColumnIndex": col_idx + 1,
                },
                "cell": {"userEnteredFormat": {
                    "textFormat": {"bold": True, "foregroundColor": RED_TEXT},
                }},
                "fields": "userEnteredFormat.textFormat(bold,foregroundColor)",
            }
        })

    return requests


# ---------------------------------------------------------------------------
# Main upload
# ---------------------------------------------------------------------------


def upload_summary(
    service, spreadsheet_id: str, data: dict,
    target_title: str = "Summary", dry_run: bool = False,
) -> None:
    rows = build_summary_rows(data)
    fmt = data.get("format", {})
    n_rows = len(rows)
    n_cols = len(rows[0])

    print(f"Target: '{target_title}' | {n_rows} rows x {n_cols} cols")

    if dry_run:
        for i, row in enumerate(rows[:5]):
            print(f"  Row {i}: {row[:6]}...")
        red_cells = find_column_max_rows(rows, data)
        print(f"  Red bold cells: {len(red_cells)}")
        return

    sheet_id = get_or_create_sheet(service, spreadsheet_id, target_title)
    print(f"  Sheet ID: {sheet_id}")

    print("  Clearing...")
    clear_sheet(service, spreadsheet_id, sheet_id, target_title)

    print(f"  Writing {n_rows} rows...")
    service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=f"{target_title}!A1",
        valueInputOption="USER_ENTERED",
        body={"values": rows},
    ).execute()

    print("  Formatting...")
    requests = build_format_requests(sheet_id, rows, fmt, data)
    service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"requests": requests},
    ).execute()
    print(f"  Done! {len(requests)} operations.")


def main():
    parser = argparse.ArgumentParser(description="Upload med-eval results to Google Sheets")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sheet", default=None, help="Override spreadsheet ID")
    parser.add_argument("--target", default="Summary", help="Target tab name")
    args = parser.parse_args()

    data = load_data(args.data)
    spreadsheet_id = args.sheet or data["spreadsheet_id"]
    total = sum(len(g["models"]) for g in data["summary"]["groups"])
    print(f"Data: {args.data} | Models: {total} | Target: {args.target}")

    if args.dry_run:
        upload_summary(None, spreadsheet_id, data, args.target, dry_run=True)
        return

    service = get_sheets_service()
    upload_summary(service, spreadsheet_id, data, args.target)
    print(f"\nhttps://docs.google.com/spreadsheets/d/{spreadsheet_id}")


if __name__ == "__main__":
    main()
