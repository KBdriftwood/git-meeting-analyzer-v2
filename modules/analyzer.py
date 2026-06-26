"""
GPTによるリアルタイム会議分析モジュール。

発言テキストを受け取り、以下を返す:
  - topic        : 発言トピック（短い要約）
  - branch       : 所属ブランチ（議題カテゴリ）
  - intent       : 発言意図リスト
  - mode         : 会議モード（報告/分析/意思決定/雑談）
  - importance   : 重要度 1-5
  - emotion      : 感情（neutral/positive/negative/urgent）
  - current_points: 現在の論点リスト
  - suggestions  : 推奨発言リスト
  - goal_progress: 会議ゴール達成確率 0-100
  - next_topics  : 次に来ると予測される論点
"""

import json
import os
from typing import Any, List, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── AIバックエンド設定 ──────────────────────────────────────────────
# "groq"    → Groq無料枠（要: GROQ_API_KEY）
# "openai"  → OpenAI API（要: OPENAI_API_KEY）
# "ollama"  → ローカル無料（要: ollama serve が起動していること）
AI_BACKEND = os.getenv("AI_BACKEND", "groq")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.1")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o")
GROQ_MODEL      = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if AI_BACKEND == "groq":
            _client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=os.getenv("GROQ_API_KEY"),
            )
        elif AI_BACKEND == "ollama":
            _client = OpenAI(
                base_url=OLLAMA_BASE_URL,
                api_key="ollama",
            )
        else:
            _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def _model_name() -> str:
    if AI_BACKEND == "groq":
        return GROQ_MODEL
    if AI_BACKEND == "ollama":
        return OLLAMA_MODEL
    return OPENAI_MODEL


SYSTEM_PROMPT = """
あなたは会議分析AIです。発言内容をリアルタイムで分析し、以下のJSON形式で返してください。

分析対象:
- topic: 発言の要点（15文字以内）
- branch: 議題ブランチ（例: main / 利益 / 顧客 / 工事 / 価格 / リスク）
- intent: 発言の裏側にある意図のリスト（最大3つ、各20文字以内）
- mode: 会議モード（"報告" | "分析" | "意思決定" | "雑談"）
- importance: 重要度 1〜5
- emotion: 感情（"neutral" | "positive" | "negative" | "urgent"）
- current_points: 現在の論点リスト（最大4つ）
- suggestions: 自分が次に発言するとよい内容リスト（最大3つ、20文字以内）
- needed_info: 今求められている情報キーワード（最大3つ）
- goal_progress: 会議ゴール達成確率 0〜100（整数）
- next_topics: 次に来ると予測される論点（最大3つ）

【重要】参加者プロフィールが提供されている場合、その人物の価値観・優先順位を踏まえて意図を深く分析してください。
表面の言葉だけでなく、その人が「なぜそれを言うのか」「何を本当に確認したいのか」を推定してください。

日本語で返してください。JSON以外のテキストは含めないでください。
""".strip()


def _build_profiles_text(profiles: List[dict]) -> str:
    """参加者プロフィールをプロンプト用テキストに変換"""
    if not profiles:
        return ""
    lines = ["【参加者プロフィール】"]
    for p in profiles:
        name = p.get("name", "")
        role = p.get("role", "")
        values = p.get("values", "")
        if name:
            lines.append(f"・{name}（{role}）: {values}")
    return "\n".join(lines)


def analyze_utterance(
    text: str,
    speaker: str,
    conversation_history: List[dict],
    meeting_goal: str = "",
    participant_profiles: List[dict] = [],
) -> dict:
    """
    1発言をGPT-4oで分析する。

    Args:
        text: 発言テキスト
        speaker: 発言者名
        conversation_history: 直近の会話履歴 [{"speaker": ..., "text": ...}]
        meeting_goal: 会議の目的（任意）

    Returns:
        分析結果の dict
    """
    client = _get_client()

    history_text = "\n".join(
        f"{u['speaker']}: {u['text']}" for u in conversation_history[-8:]
    )
    goal_text = f"\n会議目的: {meeting_goal}" if meeting_goal else ""
    profiles_text = _build_profiles_text(participant_profiles)

    user_message = f"""
{profiles_text}

会議の流れ（直近）:
{history_text}

今の発言:
{speaker}: {text}
{goal_text}

上記を分析してJSON形式で返してください。
""".strip()

    try:
        kwargs = dict(
            model=_model_name(),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=600,
            temperature=0.3,
        )
        # OllamaはJSON modeをサポートしていないモデルがあるため条件分岐
        if AI_BACKEND != "ollama":
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        raw = response.choices[0].message.content
        # JSON部分だけ抽出（Ollamaが前後に余計なテキストを出す場合の対策）
        start = raw.find("{")
        end = raw.rfind("}") + 1
        result = json.loads(raw[start:end]) if start >= 0 else {}
        return result
    except Exception as e:
        return _fallback_analysis(text, speaker, str(e))


def _fallback_analysis(text: str, speaker: str, error: str) -> dict:
    """API失敗時のフォールバック"""
    return {
        "topic": text[:15],
        "branch": "main",
        "intent": ["内容確認"],
        "mode": "報告",
        "importance": 3,
        "emotion": "neutral",
        "current_points": ["分析中..."],
        "suggestions": ["詳細を確認する"],
        "needed_info": ["情報"],
        "goal_progress": 50,
        "next_topics": ["続きを確認"],
        "_error": error,
    }


def analyze_meeting_summary(
    utterances: List[dict],
    meeting_goal: str = "",
) -> dict:
    """
    会議終了後に全体サマリーを生成する。

    Returns:
        decisions, tasks, mindmap_nodes, summary のdictを返す
    """
    client = _get_client()

    full_text = "\n".join(
        f"{u['speaker']}: {u['text']}" for u in utterances
    )

    prompt = f"""
以下の会議記録を分析し、JSON形式で返してください。

会議目的: {meeting_goal or "不明"}

会議記録:
{full_text}

出力形式:
{{
  "summary": "会議全体の要約（3文以内）",
  "decisions": [{{"content": "決定事項", "owner": "担当者", "deadline": "期限"}}],
  "tasks": [{{"task": "タスク名", "owner": "担当者", "deadline": "期限"}}],
  "mindmap_nodes": [{{"id": "ノードID", "label": "ラベル", "parent": "親ノードIDまたはnull"}}],
  "key_insights": ["インサイト1", "インサイト2"]
}}
""".strip()

    try:
        kwargs = dict(
            model=_model_name(),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.3,
        )
        if AI_BACKEND != "ollama":
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        raw = response.choices[0].message.content
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end]) if start >= 0 else {}
    except Exception as e:
        return {"_error": str(e), "summary": "分析に失敗しました", "decisions": [], "tasks": []}
