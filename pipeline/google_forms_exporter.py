"""
pipeline/google_forms_exporter.py
----------------------------------
Export MCQs → Google Forms với auto-grading (quiz mode).

Setup một lần:
  1. Vào https://console.cloud.google.com → tạo project mới
  2. Enable "Google Forms API" + "Google Drive API"
  3. Tạo OAuth 2.0 credentials → loại "Desktop app"
  4. Download → đặt tên credentials.json, để vào thư mục gốc project

Chỉ chạy được local (không dùng trên HF Spaces — OAuth cần browser cùng máy).
"""

from pathlib import Path
from typing import Optional

SCOPES = [
    "https://www.googleapis.com/auth/forms.body",
    "https://www.googleapis.com/auth/drive.file",
]

_PROJECT_ROOT = Path(__file__).parent.parent
TOKEN_PATH    = _PROJECT_ROOT / ".google_token.json"
CREDS_PATH    = _PROJECT_ROOT / "credentials.json"


# ── Auth ───────────────────────────────────────────────────────────

def has_credentials_file() -> bool:
    return CREDS_PATH.exists()


def is_authenticated() -> bool:
    if not TOKEN_PATH.exists():
        return False
    try:
        from google.oauth2.credentials import Credentials
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
        return creds.valid or bool(creds.expired and creds.refresh_token)
    except Exception:
        return False


def authenticate() -> tuple[bool, str]:
    """
    Chạy OAuth flow — mở browser để user authorize.
    Returns (success, message).
    """
    if not CREDS_PATH.exists():
        return False, (
            "Không tìm thấy credentials.json.\n"
            "Xem hướng dẫn: tạo OAuth 2.0 credentials trên Google Cloud Console "
            "→ loại Desktop app → download → đặt tên credentials.json vào thư mục project."
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
            _save_token(creds)
            return True, "Token đã được làm mới."

        flow = InstalledAppFlow.from_client_secrets_file(str(CREDS_PATH), SCOPES)
        creds = flow.run_local_server(port=0)
        _save_token(creds)
        return True, "Xác thực thành công!"

    except Exception as e:
        return False, f"Lỗi xác thực: {e}"


def _save_token(creds):
    TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")


def _get_valid_credentials():
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
            _save_token(creds)
            return creds
        return None
    except Exception:
        return None


# ── Export ─────────────────────────────────────────────────────────

def export_to_google_forms(
    mcqs: list[dict],
    title: str = "MCQ Quiz",
) -> tuple[bool, str]:
    """
    Tạo Google Form với quiz mode + auto-grading.
    Returns (success, edit_url_or_error_message).
    """
    creds = _get_valid_credentials()
    if creds is None:
        return False, "Chưa xác thực. Vui lòng kết nối Google trước."

    try:
        from googleapiclient.discovery import build

        service = build("forms", "v1", credentials=creds, cache_discovery=False)

        # Bước 1: Tạo form trống
        form = service.forms().create(body={"info": {"title": title}}).execute()
        form_id = form["formId"]

        # Bước 2: batchUpdate — enable quiz + thêm câu hỏi
        # quiz settings phải đứng đầu list
        requests = [_quiz_settings_request()]
        for idx, mcq in enumerate(mcqs):
            requests.append(_question_request(mcq, idx))

        service.forms().batchUpdate(
            formId=form_id,
            body={"requests": requests},
        ).execute()

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
