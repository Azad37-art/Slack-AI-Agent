# tools/google_sheets_tools.py

# tools/google_sheets_tools.py

import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

from config import GOOGLE_SHEET_ID, GOOGLE_SERVICE_ACCOUNT_FILE


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def get_worksheet():
    creds = Credentials.from_service_account_file(
        GOOGLE_SERVICE_ACCOUNT_FILE,
        scopes=SCOPES,
    )

    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key(GOOGLE_SHEET_ID)

    return spreadsheet.sheet1


def load_sheet() -> pd.DataFrame:
    ws = get_worksheet()
    records = ws.get_all_records()

    df = pd.DataFrame(records)
    df = df.fillna("")

    return df


def save_sheet(df: pd.DataFrame):
    ws = get_worksheet()

    df = df.fillna("")
    values = [list(df.columns)] + df.astype(str).values.tolist()

    ws.clear()
    ws.update(values)

def update_sheet_cell(row_id, field: str, new_value: str) -> dict:
    """
    Update one cell in Google Sheet by Row ID.
    """
    ws = get_worksheet()
    df = load_sheet()

    id_col = "Row" if "Row" in df.columns else df.columns[0]

    row_mask = df[id_col].astype(str) == str(row_id)

    if not row_mask.any():
        return {
            "success": False,
            "error": f"No row found with {id_col}={row_id}",
        }

    col_match = df.columns[df.columns.str.lower() == field.lower()]

    if len(col_match) == 0:
        return {
            "success": False,
            "error": f"Column '{field}' not found. Available: {list(df.columns)}",
        }

    col_name = col_match[0]

    # pandas row index
    df_index = df[row_mask].index[0]

    # Google Sheet row number:
    # +2 because row 1 is header and pandas index starts from 0
    sheet_row = int(df_index) + 2

    # Google Sheet column number:
    # +1 because Google Sheets columns start from 1
    sheet_col = list(df.columns).index(col_name) + 1

    old_value = df.loc[df_index, col_name]

    ws.update_cell(sheet_row, sheet_col, new_value)

    # Return updated row
    df.loc[df_index, col_name] = new_value
    updated_row = df.loc[df_index].to_dict()

    return {
        "success": True,
        "message": f"Updated {id_col} {row_id}: {col_name} changed from '{old_value}' to '{new_value}'",
        "row_id": row_id,
        "field": col_name,
        "old_value": str(old_value),
        "new_value": new_value,
        "updated_row": updated_row,
    }