"""The real browser path against a page served from 127.0.0.1.

No network: a local server plays Google's front page and results page. It
checks the parts the fakes cannot - typing, replacing the previous query,
pressing Enter, clearing a consent overlay and following "Next" - in an actual
Patchright browser. Skipped when Patchright or its browser is not installed.
"""

import fakegoogle
import pytest

patchright = pytest.importorskip("patchright.sync_api")

from urllib.parse import urlsplit  # noqa: E402

from google_browser_scraper import browser  # noqa: E402
from google_browser_scraper.classify import classify  # noqa: E402


@pytest.fixture
def local_google(monkeypatch):
    server, base = fakegoogle.start()
    monkeypatch.setattr(browser, "HOME_URL", base + "/")
    yield base
    server.shutdown()


def test_type_search_replace_and_paginate(local_google):
    try:
        session = browser.BrowserSession(None, headless=True).__enter__()
    except Exception as exc:  # browser binary missing on this machine
        pytest.skip(f"no Patchright browser: {exc}")
    try:
        first = session.search("swim lessons")
        assert first.consent_dismissed
        assert "q=swim+lessons" in first.url
        assert classify("https://www.google.com/search", first.html).served
        assert {"home", "type", "load", "ready", "read"} <= set(first.timings)

        # The results page keeps the last query in its box; the second query
        # must replace it, not append to it.
        second = session.search("pool near me")
        assert "q=pool+near+me" in second.url
        assert "swim" not in urlsplit(second.url).query

        nxt = session.next_page()
        assert nxt is not None and "start=10" in nxt.url
        assert {"load", "ready", "read"} <= set(nxt.timings)
    finally:
        session.close()


@pytest.mark.parametrize("front", ["/focus-thief", "/box-swap"])
def test_a_query_cut_short_while_typing_is_typed_again(local_google, monkeypatch, front):
    monkeypatch.setattr(browser, "HOME_URL", local_google + front)
    try:
        session = browser.BrowserSession(None, headless=True).__enter__()
    except Exception as exc:  # browser binary missing on this machine
        pytest.skip(f"no Patchright browser: {exc}")
    try:
        fetched = session.search("swim lessons")
        assert "q=swim+lessons" in fetched.url
    finally:
        session.close()


def test_a_page_navigating_during_the_box_check_sends_the_browser_home():
    class Navigating:
        def query_selector_all(self, selector):
            raise RuntimeError(
                "Execution context was destroyed, most likely because of a navigation"
            )

    session = browser.BrowserSession(None)
    session.page = Navigating()
    assert session._box_visible() is False
