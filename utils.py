"""Shared helpers for Spaces I/O and comments used across all app pages."""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from io import BytesIO
from typing import Optional

import boto3
import pandas as pd
import streamlit as st
from botocore.client import Config as _BotoConfig


# ── Spaces client ─────────────────────────────────────────────────────────────

def get_spaces_client():
    key    = os.environ.get("SPACES_KEY", "")
    secret = os.environ.get("SPACES_SECRET", "")
    region = os.environ.get("SPACES_REGION", "lon1").lower().strip()
    bucket = os.environ.get("SPACES_BUCKET", "")
    if not (key and secret and bucket):
        return None, ""
    client = boto3.client(
        "s3",
        region_name=region,
        endpoint_url=f"https://{region}.digitaloceanspaces.com",
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        config=_BotoConfig(signature_version="s3v4"),
    )
    return client, bucket


def load_df_from_spaces(
    file_env_var: str,
    default_filename: str,
    excel_sheet: Optional[str] = None,
) -> tuple[pd.DataFrame | None, str]:
    """Fetch a CSV or Excel file from DO Spaces. Returns (df, error_message)."""
    client, bucket = get_spaces_client()
    if client is None:
        key    = os.environ.get("SPACES_KEY", "")
        secret = os.environ.get("SPACES_SECRET", "")
        bkt    = os.environ.get("SPACES_BUCKET", "")
        missing = [n for n, v in [("SPACES_KEY", key), ("SPACES_SECRET", secret), ("SPACES_BUCKET", bkt)] if not v]
        return None, f"Missing env vars: {', '.join(missing)}"
    filename = os.environ.get(file_env_var, default_filename)
    try:
        obj  = client.get_object(Bucket=bucket, Key=filename)
        data = obj["Body"].read()
        if filename.lower().endswith((".xlsx", ".xls")):
            xl    = pd.ExcelFile(BytesIO(data))
            sheet = (
                excel_sheet
                if excel_sheet and excel_sheet in xl.sheet_names
                else xl.sheet_names[0]
            )
            return xl.parse(sheet), ""
        return pd.read_csv(BytesIO(data)), ""
    except Exception as e:
        return None, f"{filename}: {e}"


def spaces_configured() -> bool:
    return bool(
        os.environ.get("SPACES_KEY") and
        os.environ.get("SPACES_SECRET") and
        os.environ.get("SPACES_BUCKET")
    )


# ── Comments ──────────────────────────────────────────────────────────────────

def comments_key() -> str:
    return os.environ.get("SPACES_COMMENTS_FILE", "comments.json")


@st.cache_data(ttl=60, show_spinner=False)
def load_comments() -> list:
    client, bucket = get_spaces_client()
    if client is None:
        return []
    try:
        obj = client.get_object(Bucket=bucket, Key=comments_key())
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return []


def save_comments(comments: list) -> tuple[bool, str]:
    client, bucket = get_spaces_client()
    if client is None:
        return False, "Spaces client not configured"
    try:
        data = json.dumps(comments, indent=2, default=str).encode("utf-8")
        client.put_object(
            Bucket=bucket,
            Key=comments_key(),
            Body=data,
            ContentType="application/json",
        )
        load_comments.clear()
        return True, ""
    except Exception as e:
        return False, str(e)


def render_comments_section(page_label: str) -> None:
    """Render the full comments UI. page_label is stored on each comment (e.g. 'Predictions')."""
    st.divider()
    st.markdown("<div class='section-header'>💬 Comments</div>", unsafe_allow_html=True)

    if not spaces_configured():
        st.info("Comments require Spaces to be configured.")
        return

    all_comments  = load_comments()
    open_comments     = [c for c in all_comments if not c.get("resolved", False)]
    resolved_comments = [c for c in all_comments if c.get("resolved", False)]

    if open_comments:
        for c in open_comments:
            cc1, cc2 = st.columns([11, 1])
            with cc1:
                st.markdown(
                    f"**{c['author']}** "
                    f"<span style='color:var(--text-color);opacity:0.45;font-size:0.82rem'>"
                    f"{c['timestamp'][:16].replace('T', ' ')}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(c["text"])
            with cc2:
                if st.button("✓", key=f"resolve_{c['id']}", help="Mark resolved"):
                    for x in all_comments:
                        if x["id"] == c["id"]:
                            x["resolved"] = True
                    ok, err = save_comments(all_comments)
                    if ok:
                        st.rerun()
                    else:
                        st.error(f"Could not save: {err}")
    else:
        st.markdown(
            "<div style='color:var(--text-color);opacity:0.5;font-size:0.9rem'>"
            "No open comments.</div>",
            unsafe_allow_html=True,
        )

    if resolved_comments:
        with st.expander(f"Resolved ({len(resolved_comments)})"):
            for c in resolved_comments:
                st.markdown(
                    f"~~**{c['author']}**~~ "
                    f"<span style='color:var(--text-color);opacity:0.4;font-size:0.82rem'>"
                    f"{c['timestamp'][:16].replace('T', ' ')}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"~~{c['text']}~~")

    st.markdown("<br>", unsafe_allow_html=True)
    with st.form(f"comment_form_{page_label}", clear_on_submit=True):
        name = st.text_input("Your name")
        text = st.text_area("Comment", height=100)
        submitted = st.form_submit_button("Submit comment")
        if submitted:
            if not name.strip() or not text.strip():
                st.warning("Please enter both your name and a comment.")
            else:
                new_comment = {
                    "id":        str(uuid.uuid4()),
                    "timestamp": datetime.utcnow().isoformat(),
                    "author":    name.strip(),
                    "text":      text.strip(),
                    "resolved":  False,
                    "page":      page_label,
                }
                all_comments.append(new_comment)
                ok, err = save_comments(all_comments)
                if ok:
                    st.success("Comment saved.")
                    st.rerun()
                else:
                    st.error(f"Could not save comment: {err}")
