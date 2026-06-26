"""
Git風 Meeting Analyzer — Streamlit メインアプリ
「会議の議事録を作るAIではなく、会議の意図をリアルタイムに翻訳するAI」
"""

import json
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import os as _os
AI_BACKEND = _os.getenv("AI_BACKEND", "openai")

from modules.analyzer import analyze_utterance, analyze_meeting_summary
from modules.git_flow import build_svg
from modules.transcriber import parse_plaud_transcript


# ── ページ設定 ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Git風 Meeting Analyzer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS（GitHub風ダークテーマ） ───────────────────────────────────────────────

st.markdown("""
<style>
  body, .stApp { background-color: #0d1117; color: #e6edf3; }
  .stApp { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }

  /* パネルカード */
  .panel {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 16px;
    height: 100%;
  }
  .panel-title {
    font-size: 12px;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 10px;
  }

  /* 発言カード */
  .utterance-card {
    background: #21262d;
    border-radius: 6px;
    padding: 8px 10px;
    margin-bottom: 6px;
    border-left: 3px solid;
  }
  .utterance-meta {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-bottom: 3px;
    font-size: 11px;
  }
  .utterance-text { font-size: 13px; color: #e6edf3; }

  /* バッジ */
  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 10px;
    font-weight: 500;
  }
  .badge-report  { background: #21262d; color: #8b949e; border: 1px solid #30363d; }
  .badge-analyze { background: #0d419d; color: #79c0ff; border: 1px solid #1f6feb; }
  .badge-decide  { background: #3d1f00; color: #e3b341; border: 1px solid #bb8009; }
  .badge-chat    { background: #21262d; color: #8b949e; border: 1px solid #30363d; }

  /* 論点ドット */
  .point-dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #58a6ff;
    margin-right: 6px;
    vertical-align: middle;
  }

  /* 推奨発言 */
  .suggest-card {
    background: #0d1117;
    border: 1px solid #1f6feb;
    border-radius: 6px;
    padding: 7px 10px;
    margin-bottom: 5px;
    font-size: 12px;
    color: #79c0ff;
  }

  /* プログレスバー */
  .progress-bar-bg {
    background: #21262d;
    border-radius: 4px;
    height: 6px;
    margin-top: 4px;
  }
  .progress-bar-fill {
    height: 6px;
    border-radius: 4px;
    background: #3fb950;
  }

  /* ゴールフッター */
  .goal-footer {
    background: #161b22;
    border-top: 1px solid #30363d;
    padding: 10px 20px;
    display: flex;
    gap: 28px;
    align-items: center;
    font-size: 12px;
  }

  /* ライブインジケーター */
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
  .live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #3fb950;
    animation: blink 1.5s infinite;
  }

  div[data-testid="stHorizontalBlock"] > div { padding: 0 4px; }
  .stTextInput input, .stTextArea textarea, .stSelectbox select {
    background: #21262d !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
  }
  .stButton button {
    background: #238636;
    color: #fff;
    border: none;
    border-radius: 6px;
    font-size: 13px;
  }
  .stButton button:hover { background: #2ea043; }
</style>
""", unsafe_allow_html=True)

# ── セッション状態の初期化 ───────────────────────────────────────────────────

def init_state():
    defaults = {
        "utterances": [],
        "analysis_cache": {},
        "meeting_goal": "",
        "selected_node": None,
        "meeting_active": False,
        "mode": "demo",
        "next_id": 1,
        "minutes": None,
        "participant_profiles": [],
        "pending_analysis": [],   # 分析待ちの発言IDリスト
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── ヘルパー関数 ──────────────────────────────────────────────────────────────
# ※ run_pending_analysis() は関数定義後に呼ぶ

SPEAKER_COLORS = {
    "部長": "#58a6ff",
    "課長": "#f85149",
    "営業": "#3fb950",
    "自分": "#e3b341",
}
FALLBACK_COLORS = ["#a371f7", "#79c0ff", "#ffa198", "#56d364"]

def speaker_color(name: str) -> str:
    if name in SPEAKER_COLORS:
        return SPEAKER_COLORS[name]
    speakers = list(dict.fromkeys(
        u["speaker"] for u in st.session_state.utterances
    ))
    idx = speakers.index(name) if name in speakers else 0
    return FALLBACK_COLORS[idx % len(FALLBACK_COLORS)]

def mode_badge(mode: str) -> str:
    cls = {"報告": "report", "分析": "analyze", "意思決定": "decide", "雑談": "chat"}.get(mode, "report")
    return f'<span class="badge badge-{cls}">{mode}</span>'

def add_utterance(speaker: str, text: str, auto_analyze: bool = True):
    uid = st.session_state.next_id
    utt = {
        "id": uid,
        "time": uid,
        "speaker": speaker,
        "text": text,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "topic": text[:12],
        "branch": "main",
        "mode": "報告",
        "intent": [],
        "importance": 3,
        "analyzing": False,  # 分析中フラグ
    }
    st.session_state.utterances.append(utt)
    st.session_state.next_id += 1

    if auto_analyze:
        # 分析待ちキューに追加（即座には分析しない）
        st.session_state.pending_analysis.append(uid)
    return utt


def run_pending_analysis():
    """分析待ちキューを1件ずつ処理する（毎描画サイクルの先頭で呼ぶ）"""
    if not st.session_state.pending_analysis:
        return False

    uid = st.session_state.pending_analysis[0]
    utt = next((u for u in st.session_state.utterances if u["id"] == uid), None)
    if not utt:
        st.session_state.pending_analysis.pop(0)
        return False

    # 分析実行
    history = [{"speaker": u["speaker"], "text": u["text"]} for u in st.session_state.utterances]
    idx = next(i for i, u in enumerate(st.session_state.utterances) if u["id"] == uid)
    result = analyze_utterance(
        utt["text"], utt["speaker"], history[:idx],
        meeting_goal=st.session_state.meeting_goal,
        participant_profiles=st.session_state.participant_profiles,
    )
    utt.update({
        "topic":      result.get("topic", utt["topic"]),
        "branch":     result.get("branch", "main"),
        "mode":       result.get("mode", "報告"),
        "intent":     result.get("intent", []),
        "importance": result.get("importance", 3),
        "analyzing":  False,
    })
    st.session_state.analysis_cache[uid] = result
    st.session_state.pending_analysis.pop(0)
    return True  # 1件処理した

# ── デモデータ ───────────────────────────────────────────────────────────────

DEMO_DATA = [
    ("部長", "今期の売上はどうなってる？"),
    ("営業", "顧客Aの案件、来月クロージング予定です"),
    ("部長", "その案件、利益はいくら出る？"),
    ("課長", "工事側のリソースが厳しい状況です"),
    ("営業", "顧客は早期納品を強く希望しています"),
    ("部長", "価格少し調整して受注するのはどうか"),
]

DEMO_ANALYSIS = [
    {"topic": "今期売上確認", "branch": "main",    "mode": "報告",     "intent": ["現状把握", "進捗確認"], "importance": 3,
     "current_points": ["売上確認中"], "suggestions": ["売上数字を提示する"], "needed_info": ["数字"], "goal_progress": 20, "next_topics": ["利益", "案件状況"]},
    {"topic": "顧客A案件状況", "branch": "main",   "mode": "報告",     "intent": ["進捗報告", "期待値設定"], "importance": 3,
     "current_points": ["案件状況確認中", "売上確認中"], "suggestions": ["案件の詳細を説明する"], "needed_info": ["金額", "利益"], "goal_progress": 30, "next_topics": ["利益率", "案件規模"]},
    {"topic": "利益確認",    "branch": "利益",     "mode": "分析",     "intent": ["利益確認", "優先順位判断", "投資対効果"], "importance": 5,
     "current_points": ["利益確認中", "案件評価中"], "suggestions": ["利益率を伝える", "年間規模を数字で示す"], "needed_info": ["利益", "金額", "実現性"], "goal_progress": 42, "next_topics": ["工事負荷", "価格交渉"]},
    {"topic": "工事負荷懸念", "branch": "工事",    "mode": "分析",     "intent": ["リスク提示", "実現性確認", "工数見積り要求"], "importance": 4,
     "current_points": ["利益確認中", "工事リスク確認中", "案件評価中"], "suggestions": ["工事スケジュールの対策案を提示する"], "needed_info": ["工事期間", "人員"], "goal_progress": 50, "next_topics": ["工事スケジュール", "外注可否"]},
    {"topic": "顧客要望確認", "branch": "顧客",    "mode": "分析",     "intent": ["顧客優先度提示", "受注緊急性を訴える"], "importance": 4,
     "current_points": ["顧客要望確認中", "受注判断中", "工事リスク確認中"], "suggestions": ["納品時期の選択肢を提示する", "顧客の期待値を整理する"], "needed_info": ["納期", "顧客要望"], "goal_progress": 60, "next_topics": ["価格調整", "受注判断"]},
    {"topic": "価格調整提案", "branch": "価格",    "mode": "意思決定", "intent": ["受注意思表明", "価格柔軟性提示", "決断を促す"], "importance": 5,
     "current_points": ["価格検討中", "受注判断中", "利益確認中"], "suggestions": ["価格調整案を数字で提示する", "投資回収期間を示す"], "needed_info": ["利益率", "調整幅", "投資回収"], "goal_progress": 68, "next_topics": ["受注決定", "工事計画"]},
]

def load_demo():
    st.session_state.utterances = []
    st.session_state.analysis_cache = {}
    st.session_state.next_id = 1
    for i, (speaker, text) in enumerate(DEMO_DATA):
        utt = {
            "id": i + 1, "time": i + 1,
            "speaker": speaker, "text": text,
            "timestamp": f"10:{32 + i:02d}:{10 + i * 17:02d}",
            "topic": DEMO_ANALYSIS[i]["topic"],
            "branch": DEMO_ANALYSIS[i]["branch"],
            "mode": DEMO_ANALYSIS[i]["mode"],
            "intent": DEMO_ANALYSIS[i]["intent"],
            "importance": DEMO_ANALYSIS[i]["importance"],
        }
        st.session_state.utterances.append(utt)
        st.session_state.analysis_cache[i + 1] = DEMO_ANALYSIS[i]
    st.session_state.next_id = len(DEMO_DATA) + 1
    st.session_state.meeting_goal = "案件受注判断・予算承認"

# ── 分析待ちキューを処理（毎描画サイクルで1件） ───────────────────────────────
_did_analyze = run_pending_analysis()
if _did_analyze:
    st.rerun()   # 分析完了 → 画面を更新

# ── ヘッダーバー ─────────────────────────────────────────────────────────────

col_title, col_status, col_ctrl = st.columns([3, 2, 2])
with col_title:
    st.markdown(
        '<h2 style="margin:0;padding:6px 0;color:#e6edf3;font-size:20px;">'
        '🌿 Git風 Meeting Analyzer</h2>'
        '<p style="margin:0;font-size:11px;color:#8b949e;">'
        '会議の意図をリアルタイムに翻訳するAI</p>',
        unsafe_allow_html=True,
    )
with col_status:
    utterances = st.session_state.utterances
    last_analysis = (
        st.session_state.analysis_cache.get(utterances[-1]["id"], {})
        if utterances else {}
    )
    mode_now = last_analysis.get("mode", utterances[-1]["mode"] if utterances else "—")
    progress = last_analysis.get("goal_progress", 0) if utterances else 0
    st.markdown(
        f'<div style="padding:8px 0;">'
        f'<span style="font-size:11px;color:#8b949e;">会議モード: </span>'
        f'{mode_badge(mode_now)}&nbsp;&nbsp;'
        f'<span class="live-dot"></span>'
        f'<span style="font-size:11px;color:#8b949e;margin-left:6px;">LIVE</span><br>'
        f'<span style="font-size:11px;color:#8b949e;">ゴール達成: </span>'
        f'<span style="color:#3fb950;font-weight:600;">{progress}%</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
with col_ctrl:
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        if st.button("デモ表示", use_container_width=True):
            load_demo()
            st.rerun()
    with mcol2:
        if st.button("リセット", use_container_width=True):
            for k in ["utterances","analysis_cache","next_id","meeting_goal","selected_node","minutes"]:
                st.session_state[k] = [] if k in ["utterances"] else ({} if k=="analysis_cache" else (1 if k=="next_id" else (None if k in ["selected_node","minutes"] else "")))
            st.rerun()

st.markdown('<hr style="border-color:#30363d;margin:8px 0;">', unsafe_allow_html=True)

# ── サイドバー：参加者プロフィール登録 ──────────────────────────────────────

with st.sidebar:
    st.markdown("### 発言者名を変更する")
    st.markdown('<p style="font-size:12px;color:#8b949e;">会議後に名前を実名に変換できます</p>', unsafe_allow_html=True)

    # ── 発言者名の一括変更 ──────────────────────────
    st.markdown("### 発言者名を変更する")
    st.markdown('<p style="font-size:12px;color:#8b949e;">A・B・Cなどを実名に変換できます</p>', unsafe_allow_html=True)

    current_speakers = list(dict.fromkeys(
        u["speaker"] for u in st.session_state.utterances
    ))

    if current_speakers:
        rename_map = {}
        for sp in current_speakers:
            new_name = st.text_input(
                f"{sp} →",
                value=sp,
                key=f"rename_{sp}",
                placeholder="新しい名前を入力",
            )
            if new_name and new_name != sp:
                rename_map[sp] = new_name

        if st.button("名前を一括変換する", use_container_width=True):
            if rename_map:
                # 発言リストの名前を変換
                for u in st.session_state.utterances:
                    if u["speaker"] in rename_map:
                        u["speaker"] = rename_map[u["speaker"]]
                # プロフィールの名前も変換
                for p in st.session_state.participant_profiles:
                    if p["name"] in rename_map:
                        p["name"] = rename_map[p["name"]]
                st.success("変換しました")
                st.rerun()
    else:
        st.markdown('<p style="font-size:12px;color:#8b949e;">発言データがありません</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 設定")
    st.markdown(f'<p style="font-size:11px;color:#8b949e;">AIモデル: {_os.getenv("OLLAMA_MODEL","qwen2.5:14b")}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:11px;color:#8b949e;">バックエンド: {AI_BACKEND}</p>', unsafe_allow_html=True)

# ── メインレイアウト：左 + 中央 + 右 ─────────────────────────────────────────

col_left, col_center, col_right = st.columns([2.2, 3.2, 2.2], gap="small")

# ════════════════════════════════
# 左パネル: Git風フロー
# ════════════════════════════════
with col_left:
    st.markdown('<div class="panel-title">Git風 会議フロー</div>', unsafe_allow_html=True)

    node_svg = build_svg(
        utterances=st.session_state.utterances,
        selected_id=st.session_state.selected_node,
        dark_mode=True,
    )
    st.components.v1.html(
        f'<div style="overflow-x:auto;padding:4px 0;">{node_svg}</div>',
        height=300,
        scrolling=True,
    )

    # ノード選択（セレクトボックスで代替）
    if st.session_state.utterances:
        options = ["（なし）"] + [f"T{u['time']} {u['speaker']}: {u['topic']}" for u in st.session_state.utterances]
        sel_idx = st.selectbox("ノード詳細", options, key="node_select", label_visibility="collapsed")
        if sel_idx != "（なし）":
            t_num = int(sel_idx.split(" ")[0][1:])
            st.session_state.selected_node = t_num
            sel_utt = next((u for u in st.session_state.utterances if u["time"] == t_num), None)
            if sel_utt:
                analysis = st.session_state.analysis_cache.get(sel_utt["id"], {})
                st.markdown(
                    f'<div style="background:#21262d;border-radius:6px;padding:10px;margin-top:6px;">'
                    f'<div style="color:{speaker_color(sel_utt["speaker"])};font-weight:600;font-size:13px;">'
                    f'{sel_utt["speaker"]}</div>'
                    f'<div style="color:#e6edf3;font-size:13px;margin:4px 0;">{sel_utt["text"]}</div>'
                    f'<div style="color:#8b949e;font-size:11px;">ブランチ: {sel_utt["branch"]} | {mode_badge(sel_utt["mode"])}</div>'
                    + (
                        f'<div style="margin-top:8px;color:#8b949e;font-size:11px;">意図:</div>'
                        + "".join(f'<div style="font-size:12px;color:#c9d1d9;padding:2px 0;">・{i}</div>' for i in sel_utt.get("intent", []))
                        if sel_utt.get("intent") else ""
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.session_state.selected_node = None

# ════════════════════════════════
# 中央パネル: リアルタイム文字起こし
# ════════════════════════════════
with col_center:
    st.markdown('<div class="panel-title">リアルタイム文字起こし</div>', unsafe_allow_html=True)

    if not st.session_state.utterances:
        st.markdown(
            '<div style="color:#8b949e;font-size:13px;padding:20px;text-align:center;">'
            '「デモ表示」か「発言を追加」で会議を開始してください</div>',
            unsafe_allow_html=True,
        )
    else:
        html_parts = []
        for u in reversed(st.session_state.utterances[-12:]):
            color = speaker_color(u["speaker"])
            mode_b = mode_badge(u.get("mode", "報告"))
            imp = u.get("importance", 3)
            border_w = "3px" if imp >= 4 else "2px"
            is_pending = u["id"] in st.session_state.pending_analysis
            analyzing_badge = '<span style="color:#8b949e;font-size:10px;">⏳ 分析中...</span>' if is_pending else ""
            html_parts.append(
                f'<div class="utterance-card" style="border-left:{border_w} solid {color};opacity:{"0.7" if is_pending else "1"};">'
                f'<div class="utterance-meta">'
                f'<span style="color:{color};font-weight:600;">{u["speaker"]}</span>'
                f'<span style="color:#8b949e;">{u.get("timestamp","")}</span>'
                f'{mode_b}{analyzing_badge}'
                + (f'<span style="color:#e3b341;font-size:10px;">★ 重要</span>' if imp >= 5 else "")
                + f'</div>'
                f'<div class="utterance-text">{u["text"]}</div>'
                f'</div>'
            )
        st.markdown("".join(html_parts), unsafe_allow_html=True)

    st.markdown("---")

    # ── 音声入力 ──
    with st.expander("🎤 マイクで録音して追加", expanded=True):
        st.markdown(
            '<p style="font-size:12px;color:#8b949e;">'
            '① 録音ボタンを押す　② 話す　③ 停止　④ 「文字起こし＆追加」を押す</p>',
            unsafe_allow_html=True,
        )
        audio_val = st.audio_input("録音", label_visibility="collapsed", key="audio_rec")
        if audio_val:
            with st.spinner("文字起こし中..."):
                try:
                    import io, tempfile, os as _tmpos
                    from openai import OpenAI
                    client = OpenAI(api_key=_os.getenv("OPENAI_API_KEY"))
                    audio_data = audio_val.read()
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp.write(audio_data)
                        tmp_path = tmp.name
                    try:
                        with open(tmp_path, "rb") as f:
                            result = client.audio.transcriptions.create(
                                model="whisper-1",
                                file=f,
                                language="ja",
                                response_format="text",
                            )
                        recognized = result.strip() if isinstance(result, str) else ""
                    finally:
                        _tmpos.unlink(tmp_path)
                    if recognized:
                        add_utterance("発言者", recognized, auto_analyze=True)
                        st.rerun()
                    else:
                        st.warning("音声を認識できませんでした。もう一度お試しください。")
                except Exception as e:
                    st.error(f"文字起こしエラー: {e}")

    # ── 発言入力（テキストモード） ──
    with st.expander("発言を追加（テキスト入力）", expanded=not bool(st.session_state.utterances)):
        g1, g2 = st.columns([1.2, 3])
        with g1:
            speaker_in = st.text_input("発言者", value="営業", key="inp_speaker", label_visibility="collapsed",
                                       placeholder="発言者名")
        with g2:
            text_in = st.text_input("発言内容", key="inp_text", label_visibility="collapsed",
                                    placeholder="発言内容を入力してEnter...")
        has_api = (AI_BACKEND == "ollama") or bool(__import__("os").getenv("OPENAI_API_KEY"))
        st.session_state.openai_ready = has_api
        c1, c2 = st.columns(2)
        with c1:
            if st.button("追加 + AI分析" if has_api else "追加（デモ分析）", use_container_width=True):
                if text_in.strip():
                    add_utterance(speaker_in.strip() or "不明", text_in.strip(), auto_analyze=has_api)
                    if not has_api and st.session_state.utterances:
                        i = (len(st.session_state.utterances) - 1) % len(DEMO_ANALYSIS)
                        analysis = DEMO_ANALYSIS[i]
                        uid = st.session_state.utterances[-1]["id"]
                        st.session_state.utterances[-1].update({
                            "topic": analysis["topic"], "branch": analysis["branch"],
                            "mode": analysis["mode"], "intent": analysis["intent"],
                        })
                        st.session_state.analysis_cache[uid] = analysis
                    st.rerun()
        with c2:
            meeting_goal = st.text_input("会議目的", value=st.session_state.meeting_goal,
                                         key="goal_in", label_visibility="collapsed",
                                         placeholder="会議の目的（任意）")
            if meeting_goal != st.session_state.meeting_goal:
                st.session_state.meeting_goal = meeting_goal

    # ── PLAUDテキスト一括読み込み ──
    with st.expander("PLAUD / テキスト一括読み込み"):
        raw = st.text_area(
            "文字起こしテキストを貼り付け",
            height=100,
            placeholder="00:00:12 部長: 今期の売上はどうなってる？\n00:00:28 営業: 顧客Aの案件、来月クロージング予定です",
            label_visibility="collapsed",
        )
        if st.button("読み込む", use_container_width=True):
            if raw.strip():
                parsed = parse_plaud_transcript(raw)
                for p in parsed:
                    add_utterance(p["speaker"], p["text"], auto_analyze=False)
                st.rerun()

# ════════════════════════════════
# 右パネル: 3段構成
# ════════════════════════════════
with col_right:

    last = st.session_state.analysis_cache.get(
        utterances[-1]["id"] if utterances else 0, {}
    )

    # 右上: 現在の論点
    st.markdown('<div class="panel-title">現在の論点</div>', unsafe_allow_html=True)
    points = last.get("current_points", [])
    if points:
        pts_html = "".join(
            f'<div style="padding:4px 0;font-size:13px;">'
            f'<span class="point-dot"></span>{p}</div>'
            for p in points
        )
        st.markdown(pts_html, unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#8b949e;font-size:12px;">分析待ち...</span>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#30363d;margin:10px 0;">', unsafe_allow_html=True)

    # 右中: 発言意図分析
    st.markdown('<div class="panel-title">発言意図分析</div>', unsafe_allow_html=True)
    if utterances:
        last_utt = utterances[-1]
        color = speaker_color(last_utt["speaker"])
        intents = last_utt.get("intent", [])
        st.markdown(
            f'<div style="background:#21262d;border-radius:6px;padding:10px;margin-bottom:8px;">'
            f'<div style="display:flex;gap:8px;align-items:center;margin-bottom:6px;">'
            f'<span style="color:{color};font-weight:600;font-size:13px;">{last_utt["speaker"]}</span>'
            f'<span style="color:#8b949e;font-size:12px;">「{last_utt["text"][:20] + ("..." if len(last_utt["text"]) > 20 else "")}」</span>'
            f'</div>'
            + (
                '<div style="color:#8b949e;font-size:11px;margin-bottom:4px;">AI分析:</div>'
                + "".join(
                    f'<div style="font-size:12px;color:#c9d1d9;padding:2px 0;">'
                    f'<span style="color:#58a6ff;">◦</span> {i}</div>'
                    for i in intents
                ) if intents else '<div style="color:#8b949e;font-size:12px;">分析中...</div>'
            )
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<span style="color:#8b949e;font-size:12px;">発言待ち...</span>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#30363d;margin:10px 0;">', unsafe_allow_html=True)

    # 右下: 推奨発言
    st.markdown('<div class="panel-title">推奨発言</div>', unsafe_allow_html=True)
    needed = last.get("needed_info", [])
    suggestions = last.get("suggestions", [])
    if needed:
        tags = "".join(
            f'<span style="background:#0d419d;color:#79c0ff;padding:2px 8px;border-radius:12px;font-size:10px;margin-right:4px;">{n}</span>'
            for n in needed
        )
        st.markdown(
            f'<div style="margin-bottom:8px;font-size:11px;color:#8b949e;">今求められている情報:</div>{tags}',
            unsafe_allow_html=True,
        )
    if suggestions:
        s_html = "".join(f'<div class="suggest-card">→ {s}</div>' for s in suggestions)
        st.markdown(s_html, unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#8b949e;font-size:12px;">分析待ち...</span>', unsafe_allow_html=True)

# ── フッター: 会議ゴール予測 ──────────────────────────────────────────────────

st.markdown('<hr style="border-color:#30363d;margin:12px 0 8px;">', unsafe_allow_html=True)

fc1, fc2, fc3, fc4 = st.columns([2, 2, 3, 1.5])
goal_text = st.session_state.meeting_goal or "未設定"
progress = last.get("goal_progress", 0) if utterances else 0
next_topics = last.get("next_topics", []) if utterances else []

with fc1:
    st.markdown(
        f'<div style="font-size:11px;color:#8b949e;">会議目的</div>'
        f'<div style="font-size:13px;font-weight:600;color:#e6edf3;">{goal_text}</div>',
        unsafe_allow_html=True,
    )
with fc2:
    st.markdown(
        f'<div style="font-size:11px;color:#8b949e;">ゴール達成確率</div>'
        f'<div style="font-size:20px;font-weight:700;color:#3fb950;">{progress}%</div>'
        f'<div class="progress-bar-bg"><div class="progress-bar-fill" style="width:{progress}%;"></div></div>',
        unsafe_allow_html=True,
    )
with fc3:
    if next_topics:
        tags = "".join(
            f'<span style="background:#21262d;color:#8b949e;padding:2px 8px;'
            f'border-radius:12px;font-size:11px;margin-right:4px;border:1px solid #30363d;">{t}</span>'
            for t in next_topics
        )
        st.markdown(
            f'<div style="font-size:11px;color:#8b949e;">次の論点予測</div><div style="margin-top:4px;">{tags}</div>',
            unsafe_allow_html=True,
        )
with fc4:
    if utterances:
        elapsed = len(utterances)
        st.markdown(
            f'<div style="font-size:11px;color:#8b949e;text-align:right;">発言数</div>'
            f'<div style="font-size:20px;font-weight:700;color:#e6edf3;text-align:right;">{elapsed}</div>',
            unsafe_allow_html=True,
        )

# ── 会議終了セクション ────────────────────────────────────────────────────────

st.markdown('<hr style="border-color:#30363d;margin:12px 0;">', unsafe_allow_html=True)

with st.expander("会議終了 — 議事録・マインドマップを生成"):
    if st.button("AIで会議サマリーを生成する", use_container_width=True):
        if not st.session_state.utterances:
            st.warning("発言データがありません")
        else:
            with st.spinner("分析中（llama3.1で生成中...）"):
                st.session_state.minutes = analyze_meeting_summary(
                    st.session_state.utterances,
                    meeting_goal=st.session_state.meeting_goal,
                )
            st.rerun()

    if st.session_state.minutes:
        m = st.session_state.minutes
        st.markdown(f'**要約:** {m.get("summary", "")}')
        if m.get("decisions"):
            st.markdown("**決定事項**")
            rows = [[d["content"], d.get("owner", ""), d.get("deadline", "")] for d in m["decisions"]]
            st.table({"内容": [r[0] for r in rows], "担当": [r[1] for r in rows], "期限": [r[2] for r in rows]})
        if m.get("tasks"):
            st.markdown("**タスク一覧**")
            rows = [[t["task"], t.get("owner", ""), t.get("deadline", "")] for t in m["tasks"]]
            st.table({"タスク": [r[0] for r in rows], "担当": [r[1] for r in rows], "期限": [r[2] for r in rows]})
        if m.get("key_insights"):
            st.markdown("**キーインサイト**")
            for ins in m["key_insights"]:
                st.markdown(f"- {ins}")
        st.download_button(
            "議事録をJSONで保存",
            data=json.dumps(
                {"utterances": st.session_state.utterances, "summary": m},
                ensure_ascii=False, indent=2,
            ),
            file_name=f"meeting_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
        )
