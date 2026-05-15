"""
pipeline/google_forms_exporter.py
----------------------------------
Export MCQs → Google Forms với auto-grading (quiz mode).

Có 2 chế độ auth — ưu tiên theo thứ tự:

1. Service Account (HF Spaces / production)
   - Đặt biến môi trường GOOGLE_SERVICE_ACCOUNT_JSON = nội dung file JSON
   - HF Spaces: Settings → Secrets → thêm GOOGLE_SERVICE_ACCOUNT_JSON
   - Form được tạo trong Drive của service account, share public tự động

2. OAuth Desktop (local dev)
   - Tạo OAuth 2.0 credentials (Desktop app) trên Google Cloud Console
   - Download → đặt tên credentials.json vào thư mục gốc project
   - Lần đầu chạy sẽ mở browser để authorize
"""

import json
import os
from pathlib import Path
from typing import Optional

SCOPES = [
    "https://www.googleapis.com/auth/forms.body",
    "https://www.googleapis.com/auth/drive.file",
]

_PROJECT_ROOT = Path(__file__).parent.parent
TOKEN_PATH    = _PROJECT_ROOT / ".google_token.json"
CREDS_PATH    = _PROJECT_ROOT / "credentials.json"


# ── Auth mode detection ────────────────────────────────────────────

def _get_service_account_credentials():
    """Lấy credentials từ GOOGLE_SERVICE_ACCOUNT_JSON env var."""
    sa_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if not sa_json:
        return None
    try:
        from google.oauth2 import service_account
        info = json.loads(sa_json)
        return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    except Exception:
        return None


def _get_oauth_credentials():
    """Lấy credentials từ local OAuth token."""
    if not TOKEN_PATH.exists():
        return None
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
        if creds.valid:
            return creds
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
            return creds
        return None
    except Exception:
        return None


def _get_valid_credentials():
    """Service Account trước, OAuth sau."""
    return _get_service_account_credentials() or _get_oauth_credentials()


# ── Status helpers ─────────────────────────────────────────────────

def has_service_account() -> bool:
    return bool(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip())

def has_credentials_file() -> bool:
    return CREDS_PATH.exists()

def is_authenticated() -> bool:
    return _get_valid_credentials() is not None

def auth_mode() -> str:
    """'service_account' | 'oauth' | 'none'"""
    if has_service_account():
        creds = _get_service_account_credentials()
        return "service_account" if creds else "none"
    if _get_oauth_credentials():
        return "oauth"
    return "none"


# ── OAuth flow (local only) ────────────────────────────────────────

def authenticate() -> tuple[bool, str]:
    """OAuth Desktop flow — chỉ dùng local."""
    if has_service_account():
        return True, "Đang dùng Service Account — không cần kết nối thủ công."

    if not CREDS_PATH.exists():
        return False, (
            "Không tìm thấy credentials.json. "
            "Tạo OAuth 2.0 credentials (Desktop app) trên Google Cloud Console "
            "→ download → đặt vào thư mục project."
        )
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request

        creds = None
        if TOKEN_PATH.exists():
            creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

        if creds and creds.valid:
            return True, "Đã xác thực."
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
            return True, "Token đã được làm mới."

        flow = InstalledAppFlow.from_client_secrets_file(str(CREDS_PATH), SCOPES)
        creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
        return True, "Xác thực thành công!"

    except Exception as e:
        return False, f"Lỗi xác thực: {e}"


# ── Export ─────────────────────────────────────────────────────────

def export_to_google_forms(
    mcqs: list[dict],
    title: str = "MCQ Quiz",
) -> tuple[bool, str]:
    """
    Tạo Google Form với quiz mode + auto-grading.
    Returns (success, view_url_or_error).
    """
    creds = _get_valid_credentials()
    if creds is None:
        return False, "Chưa xác thực. Vui lòng kết nối Google hoặc cài GOOGLE_SERVICE_ACCOUNT_JSON."

    try:
        from googleapiclient.discovery import build

        forms_svc = build("forms", "v1", credentials=creds, cache_discovery=False)

        form = forms_svc.forms().create(body={"info": {"title": title}}).execute()
        form_id = form["formId"]

        requests = [_quiz_settings_request()]
        for idx, mcq in enumerate(mcqs):
            requests.append(_question_request(mcq, idx))

        forms_svc.forms().batchUpdate(
            formId=form_id,
            body={"requests": requests},
        ).execute()

        # Service account: share form publicly so anyone with link can view
        if has_service_account():
            try:
                drive_svc = build("drive", "v3", credentials=creds, cache_discovery=False)
                drive_svc.permissions().create(
                    fileId=form_id,
                    body={"type": "anyone", "role": "reader"},
                ).execute()
            except Exception:
                pass  # share thất bại vẫn trả link — user có thể mở nếu được share riêng

        view_url = f"https://docs.google.com/forms/d/{form_id}/viewform"
        return True, view_url

    except Exception as e:
        return False, f"Lỗi tạo form: {e}"


# ── Request builders ───────────────────────────────────────────────

def _quiz_settings_request() -> dict:
    return {
        "updateSettings": {
            "settings": {"quizSettings": {"isQuiz": True}},
            "updateMask": "quizSettings.isQuiz",
        }
    }


def _question_request(mcq: dict, index: int) -> dict:
    correct_letter = mcq.get("answer", "A")
    correct_text   = mcq.get(correct_letter, "")
    explanation    = mcq.get("explanation", "")

    options = [
        {"value": mcq.get(opt, "")}
        for opt in ["A", "B", "C", "D"]
        if mcq.get(opt)
    ]

    return {
        "createItem": {
            "item": {
                "title": mcq.get("question", ""),
                "questionItem": {
                    "question": {
                        "required": True,
                        "grading": {
                            "pointValue": 1,
                            "correctAnswers": {
                                "answers": [{"value": correct_text}]
                            },
                            "whenRight": {"text": "Chính xác!"},
                            "whenWrong": {
                                "text": (
                                    f"Đáp án đúng: {correct_letter}. {correct_text}. "
                                    f"{explanation[:120]}"
                                ).strip()
                            },
                        },
                        "choiceQuestion": {
                            "type": "RADIO",
                            "options": options,
                            "shuffle": False,
                        },
                    }
                },
            },
            "location": {"index": index},
        }
    }
