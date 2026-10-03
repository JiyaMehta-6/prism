"""PDF report generation (ReportLab).

Produces ``outputs/player_report.pdf`` containing the executive summary, the
fingerprint with confidence scores, every chart, the tactical timeline,
behavioural insights, improvement recommendations and the similarity analysis.

The exporter is defensive: missing charts are skipped rather than aborting the
document, and every section includes confidence figures so conclusions are
never presented without their evidence base.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Sequence
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from core.config import OUTPUT_DIR
from core.logger import get_logger
from core.models import FINGERPRINT_METRICS, AnalysisReport

logger = get_logger("pdf_export")

PRIMARY = colors.HexColor("#2F6FB2")
ACCENT = colors.HexColor("#E8833A")
LIGHT = colors.HexColor("#EEF3F8")
DARK = colors.HexColor("#1C2833")
SUCCESS = colors.HexColor("#3FA46A")
RISK = colors.HexColor("#C4453C")


class PdfCancelled(Exception):
    """Raised when the caller requests cancellation mid-export."""


def _esc(text: object) -> str:
    """Escape user-derived text before it enters a Paragraph (XML context)."""
    if text is None:
        return ""
    return escape(str(text))


def _styles() -> dict:
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "PrismTitle", parent=base["Title"], fontSize=26, textColor=PRIMARY,
            spaceAfter=6, alignment=TA_CENTER,
        ),
        "subtitle": ParagraphStyle(
            "PrismSub", parent=base["Normal"], fontSize=12, textColor=DARK,
            alignment=TA_CENTER, spaceAfter=18,
        ),
        "h1": ParagraphStyle(
            "PrismH1", parent=base["Heading1"], fontSize=16, textColor=PRIMARY,
            spaceBefore=14, spaceAfter=8, leading=20,
        ),
        "h2": ParagraphStyle(
            "PrismH2", parent=base["Heading2"], fontSize=12.5, textColor=DARK,
            spaceBefore=10, spaceAfter=5, leading=16,
        ),
        "body": ParagraphStyle(
            "PrismBody", parent=base["Normal"], fontSize=9.8, leading=14,
            alignment=TA_LEFT, textColor=DARK, spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "PrismSmall", parent=base["Normal"], fontSize=8.4, leading=11.5,
            textColor=colors.HexColor("#566573"), spaceAfter=4,
        ),
        "cell": ParagraphStyle(
            "PrismCell", parent=base["Normal"], fontSize=8.8, leading=11.5, textColor=DARK,
        ),
        "cellhead": ParagraphStyle(
            "PrismCellHead", parent=base["Normal"], fontSize=8.8, leading=11.5,
            textColor=colors.white, fontName="Helvetica-Bold",
        ),
    }
    return styles


def export_pdf(
    report: AnalysisReport,
    output_path: str | None = None,
    should_cancel=None,
) -> str:
    """Render ``report`` to a PDF and return the written file path.

    ``should_cancel`` is an optional zero-argument callable; when it returns
    ``True`` the export aborts with :class:`PdfCancelled` before any file is
    written.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = output_path or os.path.join(OUTPUT_DIR, "player_report.pdf")
    if not output_path.lower().endswith(".pdf"):
        output_path += ".pdf"
    output_path = os.path.abspath(output_path)
    if os.path.isdir(output_path):
        raise ValueError(f"PDF output path is a directory: {output_path}")
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    styles = _styles()

    tmp_path = output_path + ".tmp"
    doc = SimpleDocTemplate(
        tmp_path,
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title=f"PRISM Report - {report.profile.player_label}",
        author="PRISM",
    )

    story: List = []
    _cover(story, report, styles)
    _executive_summary(story, report, styles)
    story.append(PageBreak())
    _fingerprint_section(story, report, styles)
    _archetype_section(story, report, styles)
    story.append(PageBreak())
    _visual_section(story, report, styles)
    story.append(PageBreak())
    _timeline_section(story, report, styles)
    _insight_section(story, report, styles)
    story.append(PageBreak())
    _similarity_section(story, report, styles)
    _methodology(story, report, styles)

    if should_cancel is not None and should_cancel():
        raise PdfCancelled("PDF export cancelled")

    try:
        doc.build(story)
        # Atomic swap: readers only ever see a complete file.
        os.replace(tmp_path, output_path)
    except Exception:
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise
    logger.info("PDF report written to %s", output_path)
    return output_path


def _cover(story: List, report: AnalysisReport, styles: dict) -> None:
    story.append(Spacer(1, 2.2 * cm))
    story.append(Paragraph("PRISM", styles["title"]))
    story.append(
        Paragraph("Player Replay Intelligence and Strategic Modeling", styles["subtitle"])
    )
    story.append(Paragraph(
        f"Behavioural Intelligence Report: <b>{_esc(report.profile.player_label)}</b>",
        styles["subtitle"]))

    rows = [
        ["Generated", report.generated_at or datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["Videos analysed", str(report.profile.video_count)],
        ["Position samples", str(report.profile.total_samples)],
        ["Detection quality", f"{report.profile.detection_rate * 100:.0f}%"],
        ["Primary archetype", report.headline],
        ["Own base inferred", report.profile.own_base],
        ["Analysis window", "0-15 minutes (early game)"],
        ["Data source", "Gameplay video only (no APIs, no telemetry)"],
    ]
    if report.profile.champion:
        rows.insert(
            5,
            [
                "Champion identified",
                f"{report.profile.champion} "
                f"({report.profile.champion_confidence * 100:.0f}% confidence)",
            ],
        )
    table = Table(rows, colWidths=[5.2 * cm, 11.5 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), LIGHT),
                ("TEXTCOLOR", (0, 0), (0, -1), PRIMARY),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D5D8DC")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(Spacer(1, 0.6 * cm))
    story.append(table)
    story.append(Spacer(1, 0.8 * cm))
    story.append(Paragraph(
        "Every number in this report is derived exclusively from movement observed "
        "on the in-game minimap of the supplied recordings. Confidence percentages "
        "indicate how much evidence backs each conclusion.",
        styles["small"],
    ))


def _executive_summary(story: List, report: AnalysisReport, styles: dict) -> None:
    story.append(Paragraph("1. Executive Summary", styles["h1"]))
    profile = report.profile
    fingerprint = report.fingerprint

    top_insights = report.insights[:3]
    top_metrics = sorted(
        FINGERPRINT_METRICS,
        key=lambda metric: fingerprint.scores.get(metric, 0.0),
        reverse=True,
    )[:4]
    primary_region = (
        max(profile.region_distribution, key=profile.region_distribution.get)
        if profile.region_distribution
        else "n/a"
    )
    lines = [
        f"<b>{_esc(profile.player_label)}</b> presents as <b>{report.headline}</b> "
        f"based on {profile.video_count} recording(s) covering "
        f"{profile.total_samples} tracked positions.",
    ]
    if profile.champion:
        lines.append(
            f"Identified as playing <b>{_esc(profile.champion)}</b> "
            f"(portrait match confidence {profile.champion_confidence * 100:.0f}%)."
        )
    lines += [
        (
            "Dominant traits: "
            + ", ".join(
                f"{metric} {fingerprint.scores.get(metric, 0):.1f}/10"
                for metric in top_metrics
            )
            + "."
        ),
        f"Early-game rotation rate: <b>{profile.roam_rate:.2f} roams per minute</b>; "
        f"primary region: <b>{primary_region}</b>.",
    ]
    for line in lines:
        story.append(Paragraph(line, styles["body"]))

    if top_insights:
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph("Top findings", styles["h2"]))
        rows = [["Finding", "Confidence", "Action"]]
        for insight in top_insights:
            rows.append([
                Paragraph(_esc(insight.title), styles["cell"]),
                Paragraph(f"{insight.confidence * 100:.0f}%", styles["cell"]),
                Paragraph(_esc(insight.suggestion), styles["cell"]),
            ])
        story.append(_styled_table(rows, [6.4 * cm, 2.4 * cm, 7.9 * cm], styles))

    if profile.quality_notes:
        story.append(Paragraph("Data quality notes", styles["h2"]))
        for note in profile.quality_notes[:6]:
            story.append(Paragraph(f"• {_esc(note)}", styles["small"]))

    if report.sections:
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph("Behavioural analytics", styles["h2"]))
        section_rows = [["Area", "Score", "Confidence", "Summary"]]
        for name, section in report.sections.items():
            score = getattr(section, "score", None)
            conf = getattr(section, "confidence", None)
            summary = getattr(section, "summary", "")
            section_rows.append([
                name,
                f"{score:.1f} / 10" if score is not None else "-",
                f"{conf * 100:.0f}%" if conf is not None else "-",
                Paragraph(_esc(summary), styles["cell"]),
            ])
        story.append(
            _styled_table(section_rows, [3.0 * cm, 2.2 * cm, 2.4 * cm, 9.1 * cm], styles)
        )


def _fingerprint_section(story: List, report: AnalysisReport, styles: dict) -> None:
    story.append(Paragraph("2. Behavioural Fingerprint", styles["h1"]))
    story.append(Paragraph(
        "The fingerprint is the player's behavioural identity: seven axes, each "
        "scored 0-10 from movement indicators, accompanied by the confidence of "
        "the underlying measurement.",
        styles["body"],
    ))

    radar = report.chart_paths.get("fingerprint")
    if radar and os.path.isfile(radar):
        story.append(Image(radar, width=11.2 * cm, height=11.2 * cm))
        story.append(Spacer(1, 0.3 * cm))

    rows = [["Metric", "Score", "Confidence", "Evidence"]]
    for metric in FINGERPRINT_METRICS:
        rows.append([
            metric,
            f"{report.fingerprint.scores.get(metric, 0.0):.1f} / 10",
            f"{report.fingerprint.confidence.get(metric, 0.0) * 100:.0f}%",
            Paragraph(_esc(report.fingerprint.contributions.get(metric, "-")), styles["cell"]),
        ])
    story.append(_styled_table(rows, [3.0 * cm, 2.2 * cm, 2.5 * cm, 9.0 * cm], styles))

    bars = report.chart_paths.get("insights")
    if bars and os.path.isfile(bars):
        story.append(Spacer(1, 0.4 * cm))
        story.append(Image(bars, width=15.5 * cm, height=8.4 * cm))


def _archetype_section(story: List, report: AnalysisReport, styles: dict) -> None:
    story.append(Paragraph("3. Player Archetypes", styles["h1"]))
    story.append(Paragraph(
        "Archetypes are transparent rules evaluated against the fingerprint. "
        "Each label is shown with the measurements that triggered it.",
        styles["body"],
    ))
    for archetype in report.archetypes:
        block = [
            Paragraph(
                f"{_esc(archetype.label)} <font size=9 color='#7F8C8D'>(confidence "
                f"{archetype.confidence * 100:.0f}%)</font>",
                styles["h2"],
            )
        ]
        for reason in archetype.reasons:
            block.append(Paragraph(f"• {_esc(reason)}", styles["cell"]))
        block.append(Spacer(1, 0.2 * cm))
        story.append(KeepTogether(block))


def _visual_section(story: List, report: AnalysisReport, styles: dict) -> None:
    story.append(Paragraph("4. Movement Heatmaps and Region Distribution", styles["h1"]))
    heatmap = report.chart_paths.get("heatmap")
    regions = report.chart_paths.get("regions")
    if heatmap and os.path.isfile(heatmap):
        story.append(Image(heatmap, width=12.4 * cm, height=12.4 * cm))
        story.append(Paragraph(
            "Heatmap of every tracked position over the 0-15 minute window. "
            "Brighter cells indicate longer occupancy.", styles["small"],
        ))
    if regions and os.path.isfile(regions):
        story.append(Spacer(1, 0.3 * cm))
        story.append(Image(regions, width=15.5 * cm, height=8.4 * cm))

    transitions = report.chart_paths.get("transitions")
    if transitions and os.path.isfile(transitions):
        story.append(Spacer(1, 0.3 * cm))
        story.append(Image(transitions, width=12.6 * cm, height=10.6 * cm))
        story.append(Paragraph(
            "Transition matrix: how often movement flows from one region to another.",
            styles["small"],
        ))

    consistency = report.chart_paths.get("consistency")
    if consistency and os.path.isfile(consistency):
        story.append(Spacer(1, 0.3 * cm))
        story.append(Image(consistency, width=15.5 * cm, height=10.4 * cm))
        story.append(Paragraph(
            "Match-to-match variation of the core indicators - each group of bars "
            "is one recording.", styles["small"],
        ))


def _timeline_section(story: List, report: AnalysisReport, styles: dict) -> None:
    story.append(Paragraph("5. Tactical Timeline", styles["h1"]))
    story.append(Paragraph(
        "The early game is segmented and labelled with the behavioural mode best "
        "supported by the movement in that segment.",
        styles["body"],
    ))
    timeline_chart = report.chart_paths.get("timeline")
    if timeline_chart and os.path.isfile(timeline_chart):
        story.append(Image(timeline_chart, width=15.5 * cm, height=7.9 * cm))
        story.append(Spacer(1, 0.3 * cm))

    rows = [["Window", "Label", "Aggression", "Kills", "Confidence", "Summary"]]
    for window in report.timeline:
        rows.append([
            f"{window.start_label}-{window.end_label}",
            window.label,
            f"{window.aggression_index:.1f}",
            f"{window.kills}",
            f"{window.confidence * 100:.0f}%",
            Paragraph(_esc(window.summary), styles["cell"]),
        ])
    story.append(
        _styled_table(
            rows,
            [2.3 * cm, 3.0 * cm, 1.9 * cm, 1.3 * cm, 2.1 * cm, 7.0 * cm],
            styles,
        )
    )


def _insight_section(story: List, report: AnalysisReport, styles: dict) -> None:
    story.append(Paragraph("6. Behavioural Insights", styles["h1"]))
    severity_color = {"risk": RISK, "strength": SUCCESS, "info": PRIMARY}

    for index, insight in enumerate(report.insights):
        color = severity_color.get(insight.severity, PRIMARY)
        title_style = ParagraphStyle(
            f"insight-{index}", parent=styles["h2"], textColor=color
        )
        block = [
            Paragraph(
                f"{_esc(insight.title)} <font size=9>(confidence "
                f"{insight.confidence * 100:.0f}%)</font>",
                title_style,
            ),
            Paragraph(_esc(insight.detail), styles["body"]),
        ]
        if insight.evidence:
            block.append(Paragraph(
                "Evidence: " + " | ".join(_esc(item) for item in insight.evidence),
                styles["small"],
            ))
        block.append(Spacer(1, 0.25 * cm))
        story.append(KeepTogether(block))

    story.append(PageBreak())
    story.append(Paragraph("7. Improvement Recommendations", styles["h1"]))
    rows = [["#", "Finding", "Confidence", "Recommendation"]]
    for index, insight in enumerate(report.insights, start=1):
        rows.append([
            str(index),
            Paragraph(_esc(insight.title), styles["cell"]),
            Paragraph(f"{insight.confidence * 100:.0f}%", styles["cell"]),
            Paragraph(_esc(insight.suggestion), styles["cell"]),
        ])
    story.append(_styled_table(rows, [1.0 * cm, 5.4 * cm, 2.4 * cm, 8.0 * cm], styles))


def _similarity_section(story: List, report: AnalysisReport, styles: dict) -> None:
    story.append(Paragraph("8. Similarity Analysis", styles["h1"]))
    chart = report.chart_paths.get("similarity")
    if report.similarity:
        story.append(Paragraph(
            "Behavioural vectors (fingerprint + region occupancy + movement "
            "indicators) are compared with cosine similarity against previously "
            "stored profiles.",
            styles["body"],
        ))
        rows = [["Profile", "Similarity", "Interpretation"]]
        for result in report.similarity:
            rows.append([
                result.profile_name,
                f"{result.similarity * 100:.1f}%",
                Paragraph(_esc(result.note), styles["cell"]),
            ])
        story.append(_styled_table(rows, [6.0 * cm, 3.0 * cm, 7.8 * cm], styles))
        if chart and os.path.isfile(chart):
            story.append(Spacer(1, 0.4 * cm))
            story.append(Image(chart, width=15.0 * cm, height=7.5 * cm))
    else:
        story.append(Paragraph(
            "No comparable profiles are stored yet. Run PRISM on other players to "
            "build a behavioural comparison library.",
            styles["body"],
        ))
        if chart and os.path.isfile(chart):
            story.append(Image(chart, width=15.0 * cm, height=6.0 * cm))


def _methodology(story: List, report: AnalysisReport, styles: dict) -> None:
    story.append(Paragraph("9. Methodology and Limitations", styles["h1"]))
    points = [
        "<b>Input:</b> gameplay videos only (MP4/MKV/AVI/MOV). No Riot API, replay "
        "files, telemetry or manual annotations are used.",
        "<b>Position tracking:</b> the minimap is located automatically, the player "
        "marker is tracked with a background-motion + appearance model, and short "
        "gaps are interpolated.",
        "<b>Regions:</b> normalised minimap coordinates are mapped to lanes, river, "
        "jungle, bases and objective pits with an explainable geometric model.",
        "<b>OCR:</b> EasyOCR reads the match clock when possible; if confidence is "
        "too low, video time is used and the report notes the substitution.",
        "<b>Limitations:</b> vision, CS, KDA and kill events cannot be observed "
        "reliably from the minimap, so vision/objective scores are movement-based "
        "proxies with correspondingly reduced confidence.",
        "<b>Scope:</b> analysis is restricted to the first 15 minutes, where "
        "behavioural habits are most identifiable.",
    ]
    for point in points:
        story.append(Paragraph(f"• {point}", styles["body"]))

    if report.settings_snapshot:
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph("Run configuration", styles["h2"]))
        rows = [["Setting", "Value"]]
        rows.extend(
            [k.replace("_", " ").title(), str(v)]
            for k, v in report.settings_snapshot.items()
        )
        story.append(_styled_table(rows, [6.0 * cm, 10.8 * cm], styles))

    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        "PRISM is a decision-support tool for players, coaches and analysts. "
        "Insights are statistical tendencies, not guarantees about future matches.",
        styles["small"],
    ))


def _styled_table(rows: Sequence[Sequence], widths: Sequence[float], styles: dict) -> Table:
    """Append a consistently styled table (header row highlighted)."""
    header = rows[0]
    body = rows[1:]
    formatted_header = [Paragraph(str(cell), styles["cellhead"]) for cell in header]
    table = Table([formatted_header, *body], colWidths=list(widths), repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D5D8DC")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table

