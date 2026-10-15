"""The identity lifecycle: warm an exit, hold it while Google serves it, drop it
the moment Google refuses it.

The expensive event is finding an exit Google will serve. Once one is found it
usually keeps being served, so a served exit is worth a series of queries, and
a refused one is worth nothing - retrying it only burns the address further.

Traffic is counted by a loopback relay per identity (see `relay`);
`use_relay=False` hands the proxy to the browser directly instead.

With `spare` set, the next identities are warmed in the background while the
current one serves, so a retired exit is replaced without waiting for a
warm-up.

With `clones` set, the browser profile of the first exit Google serves is
copied, and the next exits start from those copies on new sticky sessions
instead of warming from scratch. Google's acceptance follows the profile more
than the address: a copy is served about as often as the profile it was taken
from, and refused when that one would be, so a copy that is refused on its
first query ends the copying.
"""

from __future__ import annotations

import gzip
import os
import random
import re
import shutil
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import browser, engines
from . import warmup as ladder
from .classify import Verdict, classify
from .links import LinkResolver
from .output import build_record
from .parse import parse_serp
from .proxy import ProxySource, new_session_id
from .relay import Relay

MB = 1024 * 1024
MAX_SPARES = 3
MAX_CLONES = 50


_CREDENTIALS = re.compile(r"//[^/\s:@]+:[^@\s]+@")
#: Chromium's own network failures: the tunnel, TLS or the connection broke.
_NETWORK_ERROR = re.compile(r"net::ERR_")


def describe_error(exc: BaseException) -> str:
    """One line naming an exception, for records and events.

    Browser errors can be long and can quote the proxy URL they were given, so
    only the first line is kept, it is truncated, and `user:pass@` is masked.
    """
    lines = str(exc).strip().splitlines()
    first = _CREDENTIALS.sub("//***@", lines[0] if lines else "")[:200]
    return f"{type(exc).__name__}: {first}" if first else type(exc).__name__


#: Chromium's single-instance locks, which must not travel with a copy.
_PROFILE_LOCKS = shutil.ignore_patterns("Singleton*", "lockfile", "*.lock")

#: Where Chromium keeps the cookie database, newest layout first.
_COOKIE_DIRS = (os.path.join("Default", "Network"), "Default")


def _retry_copy(copy, dest: str, attempts: int) -> None:
    """Run `copy` into a fresh `dest`, retrying on OSError.

    Windows can hold a file of a just-closed browser for a moment, so a failed
    copy is retried after a short wait.
    """
    for attempt in range(attempts):
        shutil.rmtree(dest, ignore_errors=True)
        try:
            copy()
            return
        except OSError:
            if attempt + 1 == attempts:
                raise
            time.sleep(1.0)


def _copy_profile(source: str, dest: str, attempts: int = 5) -> None:
    """Copy a closed browser profile into `dest` (created or replaced)."""
    _retry_copy(lambda: shutil.copytree(source, dest, ignore=_PROFILE_LOCKS), dest, attempts)


def _copy_jar(source: str, dest: str, attempts: int = 5) -> None:
    """Copy the cookie jar of a closed profile into a new profile at `dest`.

    The new profile holds the cookie database and `Local State`, which carries
    the key that decrypts it, and nothing else: storage, cache and history stay
    behind. That is a few tens of kilobytes instead of tens of megabytes.
    """
    for rel in _COOKIE_DIRS:
        folder = os.path.join(source, rel)
        if os.path.isfile(os.path.join(folder, "Cookies")):
            break
    else:
        raise RuntimeError("no cookie database in the profile")
    names = [n for n in os.listdir(folder) if n.startswith("Cookies")]

    def copy() -> None:
        os.makedirs(os.path.join(dest, rel))
        for name in names:
            shutil.copy2(os.path.join(folder, name), os.path.join(dest, rel, name))
        state = os.path.join(source, "Local State")
        if os.path.isfile(state):
            shutil.copy2(state, os.path.join(dest, "Local State"))

    _retry_copy(copy, dest, attempts)


class ExitsRefused(RuntimeError):
    """Too many consecutive exits were refused or unreachable. Raised instead
    of carrying on, because on a shared, metered pool every further identity
    costs traffic and reputation and the evidence says it will fail too."""


class ProxyRefused(ExitsRefused):
    """The proxy itself refused: no traffic left (402) or wrong credentials
    (407). No other exit through the same account can do better."""


#: Proxy answers that hold for every exit of the account, not just this one.
_ACCOUNT_REFUSALS = (402, 407)


def _account_refusal(relay: Any) -> str | None:
    """The proxy's status line if it refused for the whole account, else None."""
    refusal = getattr(relay, "refusal", None)
    if not refusal:
        return None
    parts = refusal.split(" ", 2)
    if len(parts) > 1 and parts[1].isdigit() and int(parts[1]) in _ACCOUNT_REFUSALS:
        return refusal
    return None


@dataclass
class Settings:
    engine: str = engines.DEFAULT_ENGINE
    warmup: str = ladder.DEFAULT
    pages: int = 1
    #: Each retirement throws away a warm-up worth tens of pages of traffic,
    #: and a served exit keeps being served far past ten queries.
    max_queries_per_exit: int = 50
    max_exits_per_query: int = 3
    breaker: int = 6
    gap_seconds: tuple[float, float] = (2.0, 5.0)
    dwell_seconds: tuple[float, float] = ladder.DWELL_SECONDS
    headless: bool = False
    channel: str | None = None
    timezone_id: str | None = None
    browser_args: tuple[str, ...] = ()
    use_relay: bool = True
    hl: str = "en"
    #: Directory to keep every page's HTML in, gzipped, for debugging a parse.
    #: The pages carry the exit's location, so keep them out of anything public.
    save_html: str | None = None
    #: Ask Google's /goto redirect for each result's destination, from a proxy
    #: session of its own; pages no longer carry the URLs themselves.
    resolve_links: bool = True
    #: Identities to warm in the background once the current one is served,
    #: so the next exit is ready when this one is retired. Each costs a
    #: warm-up's traffic whether it is used or not.
    spare: int = 0
    #: New exits started from copies of the first served exit's profile,
    #: instead of each warming its own. 0 turns copying off.
    clones: int = 0
    #: Seconds after which the copied profile is not used any more.
    clone_max_age: float = 3600.0
    #: Wait between relaunching a warmed profile and its first search: a
    #: search the moment the browser starts is refused far more often.
    settle_seconds: tuple[float, float] = (40.0, 60.0)

    def __post_init__(self) -> None:
        engines.get(self.engine)
        ladder.pages(self.warmup)
        for name in ("pages", "max_queries_per_exit", "max_exits_per_query", "breaker"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if not 0 <= self.spare <= MAX_SPARES:
            raise ValueError(f"spare must be between 0 and {MAX_SPARES}")
        if not 0 <= self.clones <= MAX_CLONES:
            raise ValueError(f"clones must be between 0 and {MAX_CLONES}")
        if self.clone_max_age <= 0:
            raise ValueError("clone_max_age must be positive")
        for name in ("gap_seconds", "dwell_seconds", "settle_seconds"):
            low, high = getattr(self, name)
            if not 0 <= low <= high:
                raise ValueError(f"{name} must be a (low, high) pair with 0 <= low <= high")


@dataclass
class Template:
    """A served identity's browser profile, kept on disk to start copies from."""

    path: str
    made_at: float
    left: int


@dataclass
class Identity:
    session_id: str
    session: Any
    relay: Any
    warm_pages: int
    warm_delivered: int
    warm_bytes: int | None
    queries: int = 0
    consent_dismissed: bool = False
    warm_seconds: float = 0.0
    ready_at: float = 0.0
    #: Started from a copy of another identity's profile, not warmed itself.
    clone: bool = False

    def bytes(self) -> int | None:
        return self.relay.total if self.relay is not None else None

    def info(self, level: str, page_bytes: int | None = None) -> dict[str, Any]:
        return {
            "session": self.session_id,
            "queries_on_exit": self.queries,
            "warmup": level,
            "warm_pages_delivered": f"{self.warm_delivered}/{self.warm_pages}",
            "warm_bytes": self.warm_bytes,
            "bytes": page_bytes,
            "consent_dismissed": self.consent_dismissed,
        }


@dataclass
class Stats:
    """What a run cost, counted across every identity including refused ones."""

    identities: int = 0
    clones: int = 0
    pages_served: int = 0
    queries_failed: int = 0
    exits_refused: int = 0
    exits_unreachable: int = 0
    bytes_total: int = 0
    bytes_warm: int = 0
    links_resolved: int = 0
    links_failed: int = 0
    counted: bool = True
    _open: list = field(default_factory=list, repr=False)

    def summary(self, price_per_gb: float | None = None) -> dict[str, Any]:
        total = self.bytes_total + sum(i.bytes() or 0 for i in self._open)
        out: dict[str, Any] = {
            "identities": self.identities,
            "clones": self.clones,
            "pages_served": self.pages_served,
            "queries_failed": self.queries_failed,
            "exits_refused": self.exits_refused,
            "exits_unreachable": self.exits_unreachable,
            "links_resolved": self.links_resolved,
            "links_failed": self.links_failed,
        }
        if self.counted:
            out["traffic_mb"] = round(total / MB, 2)
            out["warmup_mb"] = round(self.bytes_warm / MB, 2)
            if self.pages_served:
                per_page = total / MB / self.pages_served
                out["mb_per_served_page"] = round(per_page, 2)
                if price_per_gb is not None:
                    out["cost_per_1000_pages"] = round(per_page / 1024 * price_per_gb * 1000, 2)
        return out


class Pinned:
    """A browser session whose every call runs on one thread of its own.

    Playwright's sync API is bound to the thread that started it. A spare is
    warmed on a background thread and searched later from the caller's, so each
    spare gets a thread that owns its browser from launch to close.
    """

    def __init__(self, make: Callable[[], Any]) -> None:
        self._thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gbs-identity")
        try:
            self.session = self._call(make)
        except BaseException:
            self._thread.shutdown(wait=False)
            raise

    def _call(self, fn: Callable[..., Any], *args: Any) -> Any:
        return self._thread.submit(fn, *args).result()

    def __enter__(self) -> Pinned:
        self._call(self.session.__enter__)
        return self

    def visit(self, url: str, dwell_seconds: float) -> bool:
        return self._call(self.session.visit, url, dwell_seconds)

    def search(self, query: str) -> Any:
        return self._call(self.session.search, query)

    def next_page(self) -> Any:
        return self._call(self.session.next_page)

    def suspend(self) -> str:
        return self._call(self.session.suspend)

    def resume(self) -> None:
        self._call(self.session.resume)

    def close(self) -> None:
        try:
            self._call(self.session.close)
        finally:
            self._thread.shutdown(wait=False)


class Scraper:
    """Run queries through a pool of browser identities.

    `session_factory`, `relay_factory` and `sleep` are injectable so the
    lifecycle can be tested without a browser, a network or a clock.
    """

    def __init__(
        self,
        proxy: ProxySource | None = None,
        settings: Settings | None = None,
        *,
        session_factory: Callable[..., Any] | None = None,
        relay_factory: Callable[..., Any] | None = Relay,
        resolver_factory: Callable[..., Any] | None = LinkResolver,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.proxy = proxy
        self.settings = settings or Settings()
        self._factory = session_factory or engines.get(self.settings.engine)
        self._relay_factory = relay_factory
        self._resolver_factory = resolver_factory
        self._resolver = None
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._emit = on_event or (lambda event: None)
        self._identity: Identity | None = None
        self._spares: deque[Future] = deque()
        self._warmer: ThreadPoolExecutor | None = None
        self._cancel = threading.Event()
        self._template: Template | None = None
        self._failed_exits = 0
        self._saved = 0
        self.stats = Stats()

    # -- public --------------------------------------------------------------

    def run(self, queries: Iterable[str]) -> Iterator[dict[str, Any]]:
        """Yield one record per results page, then close the last identity."""
        try:
            pending = (q.strip() for q in queries)
            pending = (q for q in pending if q)
            query = next(pending, None)
            while query is not None:
                following = next(pending, None)
                # No spare is warmed for a run that has nothing left to ask.
                yield from self._query(query, more=following is not None)
                query = following
        finally:
            self.close()

    def search(self, query: str) -> list[dict[str, Any]]:
        """One query, keeping the identity open for the next call (serve, MCP)."""
        return list(self._query(query.strip()))

    def prewarm(self) -> None:
        """Mint and warm an identity now, so the first query does not pay for it."""
        self._current_identity()

    def reset_breaker(self) -> None:
        self._failed_exits = 0

    def close(self) -> None:
        if self._identity is not None:
            self._retire("run finished")
        self._discard_spares()
        self._drop_template("run finished")
        if self._resolver is not None:
            self._resolver.close()
            self.stats.bytes_total += self._resolver.bytes
            self.stats.links_resolved += self._resolver.resolved
            self.stats.links_failed += self._resolver.failed
            self._resolver = None

    # -- one query -----------------------------------------------------------

    def _query(self, query: str, more: bool = True) -> Iterator[dict[str, Any]]:
        s = self.settings
        last_verdict: Verdict | None = None
        last_error: str | None = None
        for _ in range(s.max_exits_per_query):
            identity = self._current_identity()
            if identity is None:
                last_error = "exit unreachable: no warm-up page arrived"
                continue
            if identity.queries:
                self._sleep(self._rng.uniform(*s.gap_seconds))
            started, before = time.monotonic(), identity.bytes()
            try:
                fetched = self._search(identity, query)
            except Exception as exc:
                last_error = describe_error(exc)
                self._check_proxy(identity)
                self._retire(f"error: {describe_error(exc)}")
                self._count_failure()
                continue
            identity.queries += 1
            identity.consent_dismissed = identity.consent_dismissed or fetched.consent_dismissed
            verdict = classify(fetched.url, fetched.html, fetched.status)
            self._event("query", identity, q=query, page=1, verdict=verdict.kind)
            saved = self._keep(fetched, query, 1, verdict)
            if verdict.served:
                self._failed_exits = 0
                if more:
                    self._top_up()
                yield self._served(query, 1, fetched, verdict, identity, started, before, saved)
                yield from self._more_pages(query, identity)
                capped = identity.queries >= s.max_queries_per_exit
                if self._identity is identity and more and self._wants_template(identity):
                    self._take_template(identity, reopen=not capped)
                if self._identity is identity and capped:
                    self._retire("query cap reached")
                return
            if identity.clone and identity.queries == 1:
                # Copies are refused together with the profile they came from.
                self._drop_template("a copy was refused on its first query")
            last_verdict, last_error = verdict, None
            self.stats.exits_refused += verdict.refused
            self._retire(verdict.kind)
            self._count_failure()
        self.stats.queries_failed += 1
        reason = last_verdict.reason if last_verdict else "unknown"
        yield build_record(
            query,
            page=1,
            verdict=last_verdict,
            url=None,
            elapsed=0.0,
            hl=s.hl,
            exit_info={"session": None},
            error=last_error or f"refused on {s.max_exits_per_query} exits: {reason}",
        )

    def _search(self, identity: Identity, query: str):
        """Search on this identity, once more from Google's front page if the
        browser fails.

        A browser error is not Google's answer: a click that timed out under an
        overlay says nothing about the exit. Retiring on it throws away a
        warm-up that costs far more than one more page.

        A network error is the exception: the proxy or the exit failed, and a
        second page through it fails the same way after another full timeout.
        """
        try:
            return identity.session.search(query)
        except Exception as exc:
            if _NETWORK_ERROR.search(str(exc)):
                raise
            self._event("search_retry", identity, error=describe_error(exc))
            identity.session.visit(browser.HOME_URL, 0)
            return identity.session.search(query)

    def _more_pages(self, query: str, identity: Identity) -> Iterator[dict[str, Any]]:
        for page in range(2, self.settings.pages + 1):
            self._sleep(self._rng.uniform(*self.settings.gap_seconds))
            started, before = time.monotonic(), identity.bytes()
            try:
                fetched = identity.session.next_page()
            except Exception as exc:
                self._retire(f"error: {describe_error(exc)}")
                yield self._failed(query, page, None, describe_error(exc), identity)
                return
            if fetched is None:
                return  # Google has no further page for this query.
            identity.queries += 1
            verdict = classify(fetched.url, fetched.html, fetched.status)
            self._event("query", identity, q=query, page=page, verdict=verdict.kind)
            saved = self._keep(fetched, query, page, verdict)
            if not verdict.served:
                self.stats.exits_refused += verdict.refused
                record = self._failed(query, page, verdict, verdict.reason, identity)
                self._retire(verdict.kind)
                yield record
                return
            yield self._served(query, page, fetched, verdict, identity, started, before, saved)

    def _links(self):
        if not self.settings.resolve_links or self._resolver_factory is None:
            return None
        if self._resolver is None:
            upstream = self.proxy.for_session(new_session_id()) if self.proxy is not None else None
            relay = self._relay_factory if self.settings.use_relay else None
            self._resolver = self._resolver_factory(upstream, relay_factory=relay)
        return self._resolver

    def _keep(self, fetched, query: str, page: int, verdict: Verdict) -> str | None:
        if not self.settings.save_html:
            return None
        folder = Path(self.settings.save_html)
        folder.mkdir(parents=True, exist_ok=True)
        self._saved += 1
        slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")[:40] or "query"
        path = folder / f"{self._saved:04d}_{verdict.kind}_{slug}_p{page}.html.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(fetched.html)
        return str(path)

    def _served(
        self, query, page, fetched, verdict, identity, started, before, saved=None
    ) -> dict[str, Any]:
        after = identity.bytes()
        page_bytes = after - before if after is not None and before is not None else None
        self.stats.pages_served += 1
        timings = dict(getattr(fetched, "timings", None) or {})
        mark = time.monotonic()
        parsed = parse_serp(fetched.html)
        timings["parse"] = round(time.monotonic() - mark, 2)
        resolver = self._links()
        if resolver is not None:
            mark = time.monotonic()
            resolver.fill(parsed["organic_results"])
            timings["links"] = round(time.monotonic() - mark, 2)
        record = build_record(
            query,
            page=page,
            verdict=verdict,
            url=fetched.url,
            hl=self.settings.hl,
            elapsed=time.monotonic() - started,
            exit_info=identity.info(self.settings.warmup, page_bytes),
            parsed=parsed,
            timings=timings,
        )
        if saved:
            record["search_metadata"]["raw_html_file"] = saved
        return record

    def _failed(self, query, page, verdict, error, identity) -> dict[str, Any]:
        return build_record(
            query,
            page=page,
            verdict=verdict,
            url=None,
            elapsed=0.0,
            hl=self.settings.hl,
            exit_info=identity.info(self.settings.warmup),
            error=error,
        )

    # -- identities ----------------------------------------------------------

    def _current_identity(self) -> Identity | None:
        if self._identity is None:
            if self._template_usable():
                self._identity = self._adopt(self._clone(), clone=True)
            elif self._spares:
                self._identity = self._take_spare()
            else:
                self._identity = self._mint()
        return self._identity

    # -- copies of a served profile ------------------------------------------

    def _wants_template(self, identity: Identity) -> bool:
        return bool(self.settings.clones) and not identity.clone and self._template is None

    def _take_template(self, identity: Identity, reopen: bool = True) -> None:
        """Copy a served identity's cookie jar, then carry on with it.

        The browser is closed for the copy: a live profile's cookie database is
        mid-write. Unless the identity is about to be retired, it is reopened
        on the same sticky session.
        """
        s = self.settings
        dest = tempfile.mkdtemp(prefix="gbs-template-")
        try:
            source = identity.session.suspend()
            _copy_jar(source, dest)
            if reopen:
                identity.session.resume()
        except Exception as exc:
            shutil.rmtree(dest, ignore_errors=True)
            self._event("template_failed", identity, error=describe_error(exc))
            self._retire(f"error: {describe_error(exc)}")
            return
        self._template = Template(dest, time.monotonic(), s.clones)
        self._event("template", identity, copies=s.clones)
        if reopen:
            self._sleep(self._rng.uniform(*s.settle_seconds))

    def _template_usable(self) -> bool:
        t = self._template
        if t is None:
            return False
        if t.left <= 0:
            self._drop_template("all copies used")
            return False
        if time.monotonic() - t.made_at > self.settings.clone_max_age:
            self._drop_template("too old")
            return False
        return True

    def _drop_template(self, reason: str) -> None:
        t, self._template = self._template, None
        if t is not None:
            shutil.rmtree(t.path, ignore_errors=True)
            self._emit({"event": "template_dropped", "reason": reason, "unused": t.left})

    def _clone(self) -> Identity:
        """A new sticky session started from a copy of the template profile."""
        s = self.settings
        t = self._template
        t.left -= 1
        profile = tempfile.mkdtemp(prefix="gbs-clone-")
        _copy_profile(t.path, profile)
        session_id = new_session_id()
        upstream = self.proxy.for_session(session_id) if self.proxy is not None else None
        relay, browser_proxy = self._relay_for(upstream)
        started = time.monotonic()
        session = self._factory(
            browser_proxy,
            headless=s.headless,
            channel=s.channel,
            timezone_id=s.timezone_id,
            browser_args=s.browser_args,
            rng=self._rng,
            profile_dir=profile,
        )
        try:
            session.__enter__()
        except Exception:
            shutil.rmtree(profile, ignore_errors=True)
            if relay is not None:
                relay.stop()
            raise
        self._sleep(self._rng.uniform(*s.settle_seconds))
        self.stats.clones += 1
        return Identity(
            session_id,
            session,
            relay,
            0,
            0,
            0,
            warm_seconds=round(time.monotonic() - started, 1),
            ready_at=time.monotonic(),
            clone=True,
        )

    def _mint(self) -> Identity | None:
        return self._adopt(self._warm(pinned=False))

    def _warm(self, pinned: bool) -> Identity:
        """Launch a browser on a new sticky session and walk the warm-up ladder.

        Touches no scraper state beyond the relay decision, so it can run on a
        background thread; the identity is counted when it is adopted.
        """
        s = self.settings
        session_id = new_session_id()
        upstream = self.proxy.for_session(session_id) if self.proxy is not None else None
        relay, browser_proxy = self._relay_for(upstream)
        started = time.monotonic()

        def make() -> Any:
            return self._factory(
                browser_proxy,
                headless=s.headless,
                channel=s.channel,
                timezone_id=s.timezone_id,
                browser_args=s.browser_args,
                rng=self._rng,
            )

        try:
            session = Pinned(make) if pinned else make()
        except Exception:
            if relay is not None:
                relay.stop()
            raise
        try:
            session.__enter__()
        except Exception:
            if pinned:
                session.close()
            if relay is not None:
                relay.stop()
            raise
        pages = ladder.pages(s.warmup)
        delivered = 0
        for url in pages:
            if pinned and self._cancel.is_set():
                break  # the run is closing; stop paying for a spare nobody will use
            delivered += bool(session.visit(url, self._rng.uniform(*s.dwell_seconds)))
            if _account_refusal(relay):
                break  # every further page is refused too, each after a full timeout
        warm_bytes = relay.total if relay is not None else None
        return Identity(
            session_id,
            session,
            relay,
            len(pages),
            delivered,
            warm_bytes,
            warm_seconds=round(time.monotonic() - started, 1),
            ready_at=time.monotonic(),
        )

    def _count_in(self, identity: Identity, **spare: Any) -> None:
        self.stats.identities += 1
        self.stats.bytes_warm += identity.warm_bytes or 0
        self.stats._open.append(identity)
        self._event(
            "minted",
            identity,
            warm_delivered=identity.warm_delivered,
            warm_pages=identity.warm_pages,
            warm_bytes=identity.warm_bytes,
            seconds=identity.warm_seconds,
            **spare,
        )

    def _adopt(self, identity: Identity, **spare: Any) -> Identity | None:
        self._count_in(identity, **spare)
        if identity.warm_pages and not identity.warm_delivered:
            # Nothing arrived: the proxy is down, the credentials are wrong or
            # the exit is dead. Not Google's verdict, but just as final.
            self._check_proxy(identity)
            self._identity = identity
            self.stats.exits_unreachable += 1
            self._retire("unreachable")
            self._count_failure()
            return None
        return identity

    def _relay_for(self, upstream: dict[str, str] | None):
        """Decide whether this identity goes through the counting relay."""
        needs = getattr(self._factory, "needs_relay", False)
        relayable = upstream is None or upstream["server"].startswith("http://")
        if self._relay_factory is None or not (self.settings.use_relay or needs):
            self.stats.counted = False
            return None, upstream
        if not relayable:
            if needs:
                raise ValueError(
                    f"the {self.settings.engine} engine needs the relay, which "
                    "speaks HTTP to the upstream proxy; use an http:// proxy URL"
                )
            self.stats.counted = False
            return None, upstream
        relay = self._relay_factory(upstream)
        return relay, {"server": relay.start()}

    # -- spares --------------------------------------------------------------

    def _top_up(self) -> None:
        """Start warming spares in the background, up to `settings.spare`."""
        if not self.settings.spare or self._template is not None:
            # A copy of the template replaces a retired exit in seconds.
            return
        if self._warmer is None:
            self._warmer = ThreadPoolExecutor(
                max_workers=self.settings.spare, thread_name_prefix="gbs-warm"
            )
        while len(self._spares) < self.settings.spare:
            self._spares.append(self._warmer.submit(self._warm, True))

    def _take_spare(self) -> Identity | None:
        """The oldest spare, waiting for its warm-up if it is still running."""
        future = self._spares.popleft()
        asked = time.monotonic()
        try:
            identity = future.result()
        except Exception as exc:
            # A spare that failed in the background is not a reason to stop
            # the run; the identity is made here instead, as without spares.
            self._emit({"event": "spare_failed", "error": describe_error(exc)})
            return self._mint()
        now = time.monotonic()
        return self._adopt(
            identity,
            spare=True,
            waited=round(now - asked, 1),
            idle=round(max(0.0, now - identity.ready_at), 1),
        )

    def _discard_spares(self) -> None:
        """Close spares nobody used. Their warm-up was paid for, so it is counted."""
        self._cancel.set()
        while self._spares:
            try:
                identity = self._spares.popleft().result()
            except Exception:
                continue
            self._count_in(identity, spare=True)
            self._close_identity(identity, "spare unused")
        if self._warmer is not None:
            self._warmer.shutdown(wait=True)
            self._warmer = None
        self._cancel = threading.Event()

    def _retire(self, reason: str) -> None:
        identity, self._identity = self._identity, None
        if identity is not None:
            self._close_identity(identity, reason)

    def _close_identity(self, identity: Identity, reason: str) -> None:
        try:
            identity.session.close()
        finally:
            if identity.relay is not None:
                identity.relay.stop()
                self.stats.bytes_total += identity.relay.total
            if identity in self.stats._open:
                self.stats._open.remove(identity)
            self._event(
                "retired", identity, reason=reason, queries=identity.queries, bytes=identity.bytes()
            )

    def _check_proxy(self, identity: Identity) -> None:
        """Stop the run if the proxy refused for the whole account.

        The browser reports any refused tunnel as `ERR_TUNNEL_CONNECTION_FAILED`;
        the relay kept the proxy's own status line. 402 and 407 will be the
        answer for every exit, so drawing more exits only adds failures.
        """
        refusal = _account_refusal(identity.relay)
        if refusal:
            if self._identity is identity:
                self._retire(f"proxy: {refusal}")
            else:
                self._close_identity(identity, f"proxy: {refusal}")
            raise ProxyRefused(
                f"the proxy refused: {refusal}. Check the account's traffic and credentials."
            )

    def _event(self, kind: str, identity: Identity, **fields) -> None:
        self._emit({"event": kind, "session": identity.session_id, **fields})

    def _count_failure(self) -> None:
        self._failed_exits += 1
        if self._failed_exits >= self.settings.breaker:
            raise ExitsRefused(
                f"{self._failed_exits} exits in a row were refused or unreachable. "
                "Stopping rather than spending more traffic: check the proxy, the "
                "machine (`google-browser-scraper doctor`) and the hour, then retry."
            )
