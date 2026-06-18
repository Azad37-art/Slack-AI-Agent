
# tools/csv_tools.py
# ─────────────────────────────────────────────────────────────
# All CSV read/write operations using pandas.
#
# KEY CHANGE: after every write, calls the targeted index update
# function instead of rebuild_index(). This means only the ONE
# changed row is re-embedded — no full rebuild, no new files.
#
#   update_field()  → calls update_row_in_index(row_id, new_row)
#   add_row()       → calls add_row_to_index(new_row)
#   delete_row()    → calls delete_row_from_index(row_id)
# ─────────────────────────────────────────────────────────────

import pandas as pd
import re
from tools.google_sheets_tools import load_sheet, save_sheet


def load_csv() -> pd.DataFrame:
    return load_sheet()


def save_csv(df: pd.DataFrame):
    save_sheet(df)


def find_similar_rows(df: pd.DataFrame, search_name: str) -> pd.DataFrame:
    """
    Search by Row, first_name, last_name, full_name, email, and all columns.
    This makes action commands work with emails too.

    Example:
    change tmccrone9@rediff.com email to new@gmail.com
    """
    search = str(search_name).strip().lower()

    if not search:
        return df.iloc[0:0]

    masks = []

    # Search Row/id column exactly
    id_col = "Row" if "Row" in df.columns else df.columns[0]
    masks.append(
        df[id_col].astype(str).str.lower().eq(search)
    )

    # Search first_name
    if "first_name" in df.columns:
        masks.append(
            df["first_name"]
            .astype(str)
            .str.lower()
            .str.contains(search, na=False, regex=False)
        )

    # Search last_name
    if "last_name" in df.columns:
        masks.append(
            df["last_name"]
            .astype(str)
            .str.lower()
            .str.contains(search, na=False, regex=False)
        )

    # Search full name
    if "first_name" in df.columns and "last_name" in df.columns:
        full_name = (
            df["first_name"].astype(str) + " " + df["last_name"].astype(str)
        ).str.lower()

        masks.append(
            full_name.str.contains(search, na=False, regex=False)
        )

    # Search email
    if "email" in df.columns:
        masks.append(
            df["email"]
            .astype(str)
            .str.lower()
            .str.contains(search, na=False, regex=False)
        )

    # Extra fallback: search all columns
    all_cols_mask = df.apply(
        lambda col: col.astype(str).str.lower().str.contains(
            search,
            na=False,
            regex=False,
        )
    ).any(axis=1)

    masks.append(all_cols_mask)

    mask = masks[0]
    for m in masks[1:]:
        mask = mask | m

    return df[mask]


def get_all_rows() -> dict:
    df = load_csv()
    return {"success": True, "data": df.to_dict(orient="records"), "total_rows": len(df)}


def get_columns() -> dict:
    df = load_csv()
    return {"columns": list(df.columns), "total_rows": len(df)}


def search_rows(query: str) -> dict:
    df   = load_csv()
    mask = df.apply(
        lambda col: col.astype(str).str.contains(query, case=False, na=False)
    ).any(axis=1)
    results = df[mask]
    return {"success": True, "data": results.to_dict(orient="records"), "matched_rows": len(results)}


# ─────────────────────────────────────────────────────────────
# WRITE OPERATIONS
# Each one saves the CSV then does a TARGETED index update.
# ─────────────────────────────────────────────────────────────

def update_field(row_id, field: str, new_value: str) -> dict:
    """
    Update one field for the row with the given Row value.
    After saving, updates only that row's embedding in Chroma.
    """
    # Import here to avoid circular import at module load time
    from agent.rag_pipeline import update_row_in_index

    df = load_csv()

    # Find the id column
    id_col   = "Row" if "Row" in df.columns else df.columns[0]
    row_mask = df[id_col].astype(str) == str(row_id)

    if not row_mask.any():
        return {"success": False, "error": f"No row found with {id_col}={row_id}"}

    # Find the target column (case-insensitive)
    col_match = df.columns[df.columns.str.lower() == field.lower()]
    if len(col_match) == 0:
        return {
            "success": False,
            "error": f"Column '{field}' not found. Available: {list(df.columns)}",
        }

    col_name = col_match[0]

    new_value = clean_slack_value(new_value)

    old_value = df.loc[row_mask, col_name].values[0]
    old_value = "" if pd.isna(old_value) else str(old_value)

    df.loc[row_mask, col_name] = new_value
    save_csv(df)

    # ── Incremental update: re-embed only this one row ────────
    updated_row = df[row_mask].to_dict(orient="records")[0]
    update_row_in_index(row_id, updated_row)

    return {
        "success":   True,
        "message":   f"Updated {id_col} {row_id}: {col_name} changed from '{old_value}' to '{new_value}'",
        "row_id":    row_id,
        "field":     col_name,
        "old_value": str(old_value),
        "new_value": new_value,
    }


def add_row(row_data: dict) -> dict:
    """
    Add a new row to the CSV.
    After saving, embeds only the new row in Chroma.
    """
    from agent.rag_pipeline import add_row_to_index

    df = load_csv()

    # Auto-generate Row id if not provided
    id_col = "Row" if "Row" in df.columns else df.columns[0]
    if id_col not in row_data:
        max_id = pd.to_numeric(df[id_col], errors="coerce").max()
        row_data[id_col] = int(max_id) + 1 if not pd.isna(max_id) else 1

    new_df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
    save_csv(new_df)

    # ── Incremental update: embed only the new row ────────────
    add_row_to_index(row_data)

    return {
        "success": True,
        "message": f"Added new row with {id_col}={row_data[id_col]}",
        "new_row": row_data,
    }


def delete_row(row_id) -> dict:
    """
    Delete the row with the given Row value from the CSV.
    After saving, removes only that row's embedding from Chroma.
    """
    from agent.rag_pipeline import delete_row_from_index

    df     = load_csv()
    id_col = "Row" if "Row" in df.columns else df.columns[0]
    mask   = df[id_col].astype(str) == str(row_id)

    if not mask.any():
        return {"success": False, "error": f"No row found with {id_col}={row_id}"}

    deleted = df[mask].to_dict(orient="records")[0]
    df      = df[~mask].reset_index(drop=True)
    save_csv(df)

    # ── Incremental update: remove only this row's embedding ──
    delete_row_from_index(row_id)

    return {
        "success":     True,
        "message":     f"Deleted row with {id_col}={row_id}",
        "deleted_row": deleted,
    }




def clean_slack_value(value) -> str:
    if value is None:
        return ""

    value = str(value).strip()

    # <mailto:test@gmail.com|test@gmail.com>
    value = re.sub(r"<mailto:([^|>]+)\|[^>]+>", r"\1", value)

    # mailto:test@gmail.com|test@gmail.com
    value = re.sub(r"mailto:([^|\s>]+)\|[^\s>]+", r"\1", value)

    # <mailto:test@gmail.com>
    value = re.sub(r"<mailto:([^>]+)>", r"\1", value)

    # <url|label>
    value = re.sub(r"<([^|>]+)\|([^>]+)>", r"\2", value)

    if value.lower() in ["none", "null", "empty", "blank", "remove", "delete", "clear", "nan"]:
        return ""

    return value