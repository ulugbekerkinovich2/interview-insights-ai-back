import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
TEST_DB_PATH = PROJECT_ROOT / "tmp_backend_test.sqlite3"

os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import database  # noqa: E402
import logic  # noqa: E402
import main  # noqa: E402


class BackendApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()
        database.init_db()
        cls.client = TestClient(main.app)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        database.engine.dispose()
        if TEST_DB_PATH.exists():
            TEST_DB_PATH.unlink()

    def setUp(self):
        with database.SessionLocal() as db:
            db.query(database.VisualRecord).delete()
            db.query(database.ChatMessage).delete()
            db.query(database.GlobalSetting).delete()
            db.query(database.Candidate).delete()
            db.query(database.User).delete()
            db.commit()

    def create_candidate(self, name="Test Candidate", answers=None):
        token = self.create_user_token()
        response = self.client.post(
            "/candidates/",
            json={
                "name": name,
                "summary": "Smoke summary",
                "status": "interview_started",
                "answers": answers or [],
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("pin", payload)
        self.assertEqual(len(payload["pin"]), 6)
        self.assertTrue(payload["pin"].isdigit())
        return payload

    def create_user_token(self, email="admin@example.com", role="SuperAdmin"):
        with database.SessionLocal() as db:
            user = db.query(database.User).filter(database.User.email == email).first()
            if user is None:
                user = database.User(
                    name="Test Admin",
                    email=email,
                    password=main.get_password_hash("admin123"),
                    role=role,
                )
                db.add(user)
                db.commit()
                db.refresh(user)

        return main.create_access_token({"sub": email, "role": role})

    def test_missing_resources_return_404(self):
        candidate_response = self.client.get("/candidates/999")
        self.assertEqual(candidate_response.status_code, 404)
        self.assertEqual(candidate_response.json()["detail"], "Candidate not found")

        visual_response = self.client.get("/candidates/999/visual")
        self.assertEqual(visual_response.status_code, 404)
        self.assertEqual(visual_response.json()["detail"], "Candidate not found")

        setting_response = self.client.get("/settings/missing_key")
        self.assertEqual(setting_response.status_code, 404)
        self.assertEqual(setting_response.json()["detail"], "Setting not found")

    def test_visual_records_return_empty_list_for_existing_candidate(self):
        candidate = self.create_candidate()

        response = self.client.get(f"/candidates/{candidate['id']}/visual")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    def test_health_reports_database_status(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["database"]["available"])
        self.assertEqual(payload["database"]["dialect"], "sqlite")

    def test_health_returns_503_when_database_is_unavailable(self):
        with patch.object(main.database, "check_database_connection", side_effect=RuntimeError("db down")):
            response = self.client.get("/health")

        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertEqual(payload["status"], "degraded")
        self.assertFalse(payload["database"]["available"])
        self.assertIn("db down", payload["database"]["detail"])

    def test_analyze_returns_502_when_ai_fails(self):
        with patch.object(main.logic, "analyze_answer", side_effect=logic.AIServiceError("ollama offline")):
            response = self.client.post("/logic/analyze/?question=Q&answer=A")

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["detail"], "ollama offline")

    def test_summary_returns_502_when_ai_fails(self):
        candidate = self.create_candidate(
            answers=[{"question": "Q1", "answer": "A1", "ai": "I1"}],
        )

        with patch.object(main.logic, "build_interview_summary", side_effect=logic.AIServiceError("summary failed")):
            response = self.client.post(f"/logic/summary/?candidate_id={candidate['id']}")

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["detail"], "summary failed")

    def test_transcribe_returns_400_for_invalid_audio(self):
        with patch.object(main.logic, "transcribe_audio", side_effect=logic.TranscriptionError("Invalid audio file")):
            response = self.client.post(
                "/logic/transcribe/",
                files={"file": ("broken.wav", b"not audio", "audio/wav")},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Invalid audio file")

    def test_process_turn_persists_result_for_candidate(self):
        candidate = self.create_candidate()
        mocked_turn = {
            "question": "Tell me about yourself",
            "answer": "I build APIs",
            "ai": "Strong backend signal",
            "voice_raw": "steady",
            "candidate_raw": "good",
        }

        with patch.object(main.logic, "process_interview_turn", return_value=mocked_turn), patch.object(main, "send_telegram_notification"):
            response = self.client.post(
                "/logic/process-turn/",
                data={
                    "candidate_id": str(candidate["id"]),
                    "question": mocked_turn["question"],
                },
                files={"file": ("answer.wav", b"fake wav", "audio/wav")},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["audio_url"].startswith("/media/audio/"))

        with database.SessionLocal() as db:
            saved_candidate = db.query(database.Candidate).filter(database.Candidate.id == candidate["id"]).first()

        self.assertIsNotNone(saved_candidate)
        self.assertEqual(len(saved_candidate.answers), 1)
        self.assertEqual(saved_candidate.answers[0]["answer"], mocked_turn["answer"])
        self.assertEqual(saved_candidate.answers[0]["audio_url"], payload["audio_url"])

    def test_access_code_generation_retries_collisions(self):
        with database.SessionLocal() as db:
            existing = database.Candidate(
                name="Existing",
                summary="",
                status="interview_started",
                access_code="111111",
                answers=[],
            )
            db.add(existing)
            db.commit()

            with patch.object(main.secrets, "choice", side_effect=list("111111222222")):
                code = main.generate_unique_access_code(db)

        self.assertEqual(code, "222222")

    def test_register_rejects_weak_password(self):
        response = self.client.post(
            "/users/register",
            data={
                "name": "Weak User",
                "email": "weak@example.com",
                "password": "123",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Parol kamida 8 ta belgidan iborat bo'lishi kerak")

    def test_register_creates_recruiter_with_hashed_password(self):
        response = self.client.post(
            "/users/register",
            data={
                "name": "Strong User",
                "email": "strong@example.com",
                "password": "strong123",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["email"], "strong@example.com")
        self.assertEqual(payload["role"], "Recruiter")
        self.assertNotIn("password", payload)

        with database.SessionLocal() as db:
            saved_user = db.query(database.User).filter(database.User.email == "strong@example.com").first()

        self.assertIsNotNone(saved_user)
        self.assertNotEqual(saved_user.password, "strong123")
        self.assertTrue(main.verify_password("strong123", saved_user.password))

    def test_register_returns_duplicate_email_detail(self):
        self.client.post(
            "/users/register",
            data={
                "name": "Strong User",
                "email": "duplicate@example.com",
                "password": "strong123",
            },
        )

        response = self.client.post(
            "/users/register",
            data={
                "name": "Duplicate User",
                "email": "duplicate@example.com",
                "password": "strong123",
            },
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"], "Bu email allaqachon ro'yxatdan o'tgan")

    def test_register_returns_backend_error_detail(self):
        with patch.object(main, "get_password_hash", side_effect=RuntimeError("hash backend unavailable")):
            response = self.client.post(
                "/users/register",
                data={
                    "name": "Broken User",
                    "email": "broken@example.com",
                    "password": "strong123",
                },
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json()["detail"],
            "Ro'yxatdan o'tishda backend xatosi: hash backend unavailable",
        )


if __name__ == "__main__":
    unittest.main()
