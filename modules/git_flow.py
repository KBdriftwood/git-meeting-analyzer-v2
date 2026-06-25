"""
Git風会議フローのSVG生成モジュール。

発言リストを受け取り、ブランチ・タイムライン形式のSVG文字列を返す。
"""

from typing import Any, List, Optional
import html as _html

# ブランチごとの固定カラー（追加された順に割り当て）
BRANCH_PALETTE = [
    "#599CE7",  # main: blue
    "#3FA266",  # branch-1: green
    "#C85898",  # branch-2: pink
    "#F0A040",  # branch-3: orange
    "#7B64B8",  # branch-4: purple
    "#2A9A8A",  # branch-5: teal
    "#C06028",  # branch-6: deep orange
]

SPEAKER_PALETTE = [
    "#599CE7", "#3FA266", "#C85898",
    "#F0A040", "#7B64B8", "#2A9A8A",
]

NODE_R = 12
H_GAP = 110   # 水平方向の発言間隔
V_GAP = 58    # ブランチ間の垂直間隔
PAD_LEFT = 72
PAD_TOP = 36
PAD_BOTTOM = 28


def build_svg(
    utterances: List[dict],
    selected_id: Optional[int] = None,
    dark_mode: bool = True,
) -> str:
    """
    utterances: [{"id": int, "time": int, "speaker": str, "topic": str, "branch": str}, ...]
    selected_id: 選択されているノードID（ハイライト表示）
    dark_mode: 背景・テキストカラーの切り替え
    """
    if not utterances:
        return _empty_svg(dark_mode)

    bg_color      = "#181818" if dark_mode else "#FCFCFC"
    text_color    = "#E4E4E4EB" if dark_mode else "#141414F0"
    sub_color     = "#E4E4E45E" if dark_mode else "#1414148A"
    stroke_color  = "#E4E4E433" if dark_mode else "#14141433"

    # ブランチ→Y座標マッピング（出現順に割り当て）
    branch_order: list[str] = []
    for u in utterances:
        b = u.get("branch", "main")
        if b not in branch_order:
            branch_order.append(b)

    branch_y = {b: PAD_TOP + i * V_GAP for i, b in enumerate(branch_order)}
    branch_color = {b: BRANCH_PALETTE[i % len(BRANCH_PALETTE)] for i, b in enumerate(branch_order)}

    # 発言者→カラーマッピング
    speakers: list[str] = []
    for u in utterances:
        s = u.get("speaker", "?")
        if s not in speakers:
            speakers.append(s)
    speaker_color = {s: SPEAKER_PALETTE[i % len(SPEAKER_PALETTE)] for i, s in enumerate(speakers)}

    # X座標: 発言順（timeキー or indexを使用）
    node_x = {u["id"]: PAD_LEFT + idx * H_GAP for idx, u in enumerate(utterances)}

    svg_w = PAD_LEFT + len(utterances) * H_GAP + 40
    svg_h = PAD_TOP + len(branch_order) * V_GAP + PAD_BOTTOM + 30

    lines: list[str] = []
    lines.append(
        f'<svg width="{svg_w}" height="{svg_h}" xmlns="http://www.w3.org/2000/svg" '
        f'style="background:{bg_color};border-radius:8px;display:block;">'
    )

    # ── ブランチラベル（左端） ──
    for branch, y in branch_y.items():
        color = branch_color[branch]
        safe = _html.escape(branch)
        lines.append(
            f'<text x="4" y="{y + 4}" fill="{color}" font-size="10" '
            f'font-family="monospace" opacity="0.8">{safe}</text>'
        )

    # ── ブランチレール（水平線） ──
    branch_nodes: dict[str, list[dict]] = {b: [] for b in branch_order}
    for u in utterances:
        branch_nodes[u.get("branch", "main")].append(u)

    for branch, nodes in branch_nodes.items():
        if len(nodes) < 1:
            continue
        y = branch_y[branch]
        color = branch_color[branch]
        xs = sorted(node_x[n["id"]] for n in nodes)
        lines.append(
            f'<line x1="{xs[0]}" y1="{y}" x2="{xs[-1]}" y2="{y}" '
            f'stroke="{color}" stroke-width="2" opacity="0.25"/>'
        )

    # ── ブランチ分岐ライン ──
    for i, u in enumerate(utterances):
        branch = u.get("branch", "main")
        if branch == "main" or branch == branch_order[0]:
            continue
        parent_branch = branch_order[0]
        # 同じブランチの最初のノードの直前の「main」ノードを探す
        branch_first_time = min(
            uu["time"] for uu in utterances if uu.get("branch") == branch
        )
        main_before = [
            uu for uu in utterances
            if uu.get("branch") == parent_branch and uu["time"] < branch_first_time
        ]
        if not main_before:
            continue
        parent_node = max(main_before, key=lambda n: n["time"])
        if parent_node["id"] == u["id"]:
            continue
        # ブランチ最初のノードのみ描画
        first_branch_node = min(
            [uu for uu in utterances if uu.get("branch") == branch],
            key=lambda n: n["time"],
        )
        if u["id"] != first_branch_node["id"]:
            continue
        x1, y1 = node_x[parent_node["id"]], branch_y[parent_branch]
        x2, y2 = node_x[u["id"]], branch_y[branch]
        color = branch_color[branch]
        lines.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="2" opacity="0.55"/>'
        )

    # ── ノード ──
    for u in utterances:
        uid = u["id"]
        cx = node_x[uid]
        branch = u.get("branch", "main")
        cy = branch_y[branch]
        speaker = u.get("speaker", "?")
        sc = speaker_color.get(speaker, SPEAKER_PALETTE[0])
        is_sel = (uid == selected_id)
        topic = u.get("topic", "")[:10]
        label = speaker[0] if speaker else "?"

        if is_sel:
            lines.append(
                f'<circle cx="{cx}" cy="{cy}" r="{NODE_R + 7}" '
                f'fill="{sc}" opacity="0.15"/>'
            )
        fill = sc if is_sel else bg_color
        stroke = sc
        text_fill = "#191c22" if is_sel else sc
        lines.append(
            f'<circle cx="{cx}" cy="{cy}" r="{NODE_R}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if not is_sel else 0}"/>'
        )
        lines.append(
            f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" '
            f'fill="{text_fill}" font-size="10" font-weight="700">{label}</text>'
        )
        safe_topic = _html.escape(topic)
        lines.append(
            f'<text x="{cx}" y="{cy + NODE_R + 14}" text-anchor="middle" '
            f'fill="{sub_color}" font-size="8.5">{safe_topic}</text>'
        )
        lines.append(
            f'<text x="{cx}" y="{cy - NODE_R - 5}" text-anchor="middle" '
            f'fill="{sub_color}" font-size="8" opacity="0.6">T{u.get("time", uid)}</text>'
        )

    # ── 発言者凡例（右下） ──
    legend_x = 4
    legend_y = svg_h - PAD_BOTTOM - len(speakers) * 14 + 4
    for i, s in enumerate(speakers):
        color = speaker_color[s]
        safe_s = _html.escape(s)
        lines.append(
            f'<circle cx="{legend_x + 5}" cy="{legend_y + i * 14}" r="4" fill="{color}"/>'
        )
        lines.append(
            f'<text x="{legend_x + 13}" y="{legend_y + i * 14 + 4}" '
            f'fill="{sub_color}" font-size="9">{safe_s}</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


def _empty_svg(dark_mode: bool) -> str:
    bg = "#181818" if dark_mode else "#FCFCFC"
    tc = "#E4E4E45E" if dark_mode else "#1414148A"
    return (
        f'<svg width="400" height="80" xmlns="http://www.w3.org/2000/svg" '
        f'style="background:{bg};border-radius:8px;display:block;">'
        f'<text x="200" y="44" text-anchor="middle" fill="{tc}" font-size="13">'
        f'会議開始を待っています...</text></svg>'
    )
