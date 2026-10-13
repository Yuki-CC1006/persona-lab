"""Accessibility guards: landmarks, names, focus, contrast, text alternatives.

These checks are structural, not a substitute for a manual keyboard pass — but
they fail loudly when a page loses its skip link, a control loses its name, or
a colour drops below the readable contrast ratio.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

STYLE_PATH = Path(__file__).resolve().parents[1] / "app" / "static" / "style.css"
CONTROLS = {"input", "select", "textarea", "button"}


class PageAudit(HTMLParser):
    """Collect the structures that decide whether a page is usable by keyboard."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.label_depth = 0
        self.label_targets: set[str] = set()
        self.controls: list[dict] = []
        self.links: list[dict] = []
        self.aria_current = 0
        self.main_id = ""
        self.skip_href = ""
        self.nav_labels: list[str] = []
        self._button_stack: list[int] = []
        self._link_stack: list[int] = []
        self._hidden_depth = 0
        self._stack: list[tuple[str, bool]] = []
        self._heading_levels: list[int] = []

    def handle_starttag(self, tag, attrs):
        attr = {k: (v or "") for k, v in attrs}
        hidden = attr.get("aria-hidden") == "true"
        self._stack.append((tag, hidden))
        if hidden:
            self._hidden_depth += 1
        if tag == "label":
            self.label_depth += 1
            if attr.get("for"):
                self.label_targets.add(attr["for"])
        if tag == "main" and attr.get("id"):
            self.main_id = attr["id"]
        if attr.get("aria-current") == "page":
            self.aria_current += 1
        if tag == "nav" and attr.get("aria-label"):
            self.nav_labels.append(attr["aria-label"])
        if tag == "a" and attr.get("class", "").startswith("skip-link"):
            self.skip_href = attr.get("href", "")
        if tag in CONTROLS:
            if attr.get("type") == "hidden":
                return
            self.controls.append(
                {
                    "tag": tag,
                    "type": attr.get("type", ""),
                    "name": attr.get("name", ""),
                    "aria": attr.get("aria-label", ""),
                    "labelledby": attr.get("aria-labelledby", ""),
                    "title": attr.get("title", ""),
                    "id": attr.get("id", ""),
                    "in_label": self.label_depth > 0,
                    "text": "",
                    "visible": self._hidden_depth == 0 and "hidden" not in attr,
                }
            )
            if tag == "button":
                self._button_stack.append(len(self.controls) - 1)
        if tag == "a":
            self.links.append(
                {
                    "href": attr.get("href", ""),
                    "aria": attr.get("aria-label", ""),
                    "title": attr.get("title", ""),
                    "text": "",
                    "visible": self._hidden_depth == 0,
                }
            )
            self._link_stack.append(len(self.links) - 1)
        if re.fullmatch(r"h[1-6]", tag):
            self._heading_levels.append(int(tag[1]))

    def handle_endtag(self, tag):
        while self._stack:
            opened, hidden = self._stack.pop()
            if hidden:
                self._hidden_depth = max(0, self._hidden_depth - 1)
            if opened == tag:
                break
        if tag == "label":
            self.label_depth = max(0, self.label_depth - 1)
        if tag == "button" and self._button_stack:
            self._button_stack.pop()
        if tag == "a" and self._link_stack:
            self._link_stack.pop()

    def handle_data(self, data):
        text = data.strip()
        if not text or self._hidden_depth:
            return
        if self._button_stack:
            self.controls[self._button_stack[-1]]["text"] += text
        if self._link_stack:
            self.links[self._link_stack[-1]]["text"] += text


def audit(html: str) -> PageAudit:
    parser = PageAudit()
    parser.feed(html)
    parser.close()
    return parser


PAGES = ["/", "/memories", "/search", "/settings", "/imports", "/diary", "/graph", "/settings/backup"]


@pytest.fixture()
def page(client: TestClient):
    def _get(path: str) -> str:
        response = client.get(path)
        assert response.status_code == 200, path
        return response.text

    return _get


@pytest.mark.parametrize("path", PAGES)
def test_every_page_has_landmarks_and_skip_link(path, page):
    result = audit(page(path))
    assert result.main_id == "main"
    assert result.skip_href == "#main"
    assert "Main navigation" in result.nav_labels


NAV_PAGES = {
    "/profile": "/profile",
    "/search": "/search",
    "/diary": "/diary",
    "/imports": "/imports",
    "/memories": "/memories",
    "/memories/candidates": "/memories/candidates",
}


@pytest.mark.parametrize("path", PAGES)
def test_active_navigation_is_announced(path, page):
    result = audit(page(path))
    # Settings lives behind the gear and the graph is an advanced view reached
    # from the profile page, so neither announces an active nav item.
    expected = 1 if path in NAV_PAGES else 0
    assert result.aria_current == expected


@pytest.mark.parametrize("path", PAGES)
def test_every_visible_control_has_an_accessible_name(path, page):
    result = audit(page(path))
    unnamed = [
        control
        for control in result.controls
        if control["visible"]
        and not (
            control["aria"]
            or control["labelledby"]
            or control["title"]
            or control["in_label"]
            or control["id"] in result.label_targets
            or (control["tag"] == "button" and control["text"])
        )
    ]
    assert unnamed == [], f"{path} has controls without an accessible name: {unnamed}"


@pytest.mark.parametrize("path", PAGES)
def test_links_have_readable_text(path, page):
    result = audit(page(path))
    nameless = [
        link
        for link in result.links
        if link["visible"] and not (link["text"] or link["aria"] or link["title"])
    ]
    assert nameless == [], f"{path} has links without a name: {nameless}"


def test_focus_ring_is_never_removed_without_replacement():
    css = STYLE_PATH.read_text(encoding="utf-8")
    assert ":focus-visible" in css
    assert "outline: 2px solid var(--accent)" in css
    # An `outline: none` without a matching focus style would hide keyboard focus.
    stripped = re.sub(r"[^{}]*:focus-visible[^{}]*\{[^}]*\}", "", css)
    assert "outline: none" not in stripped


def _root_tokens() -> dict[str, str]:
    css = STYLE_PATH.read_text(encoding="utf-8")
    root = re.search(r":root\s*\{(.*?)\}", css, re.S)
    assert root, "style.css is missing the :root variables"
    return dict(re.findall(r"--([a-z-]+):\s*(#[0-9a-fA-F]{6})", root.group(1)))


def _luminance(color: str) -> float:
    raw = color.lstrip("#")
    channels = [int(raw[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(foreground: str, background: str) -> float:
    high, low = sorted((_luminance(foreground), _luminance(background)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def test_text_colours_meet_contrast_minimum():
    tokens = _root_tokens()
    backgrounds = ["bg", "bg-elevated", "sidebar", "input"]
    foregrounds = ["ink", "muted", "accent", "green", "warn", "danger"]
    for background in backgrounds:
        for foreground in foregrounds:
            ratio = _contrast(tokens[foreground], tokens[background])
            assert ratio >= 4.5, f"{foreground} on {background} is only {ratio:.2f}:1"


def test_graph_page_offers_a_text_equivalent(client: TestClient, db):
    from app.services import memory as memory_svc

    memory_svc.create_manual_memory(db, "Alex dislikes cilantro", memory_type="aversion", subject="Alex")
    memory_svc.create_manual_memory(db, "Alex likes mildly spicy hotpot", memory_type="preference", subject="Alex")

    from app.services.profile import build_memory_graph, graph_text_outline
    from app.db.session import get_session_factory

    session = get_session_factory()()
    try:
        outline = graph_text_outline(build_memory_graph(session))
    finally:
        session.close()

    assert outline["subject"] == "Alex"
    assert {group["label"] for group in outline["groups"]} >= {"Preference", "Aversion"}
    texts = [item["text"] for group in outline["groups"] for item in group["entries"]]
    assert any("cilantro" in text for text in texts)

    page = client.get("/graph").text
    assert "Text version of the graph" in page
    assert "Alex dislikes cilantro" in page


def test_graph_outline_is_empty_safe():
    from app.services.profile import graph_text_outline

    outline = graph_text_outline({"nodes": [], "links": []})
    assert outline == {"subject": "Subject", "groups": [], "relations": []}


def test_candidate_checkbox_is_named(client: TestClient, db):
    from app.db.models import Memory, MemoryStatus

    db.add(
        Memory(
            content="Alex dislikes cilantro",
            memory_type="aversion",
            status=MemoryStatus.CANDIDATE.value,
            subject="Alex",
        )
    )
    db.commit()
    page = client.get("/memories/candidates").text
    assert "aria-label=\"Select candidate memory" in page
