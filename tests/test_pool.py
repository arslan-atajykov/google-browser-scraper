"""Several scrapers sharing one queue of queries, with fake browsers."""

import threading
import time

import pytest
from test_scraper import REFUSED, SERVED, FakeRelay, FakeSession

from google_browser_scraper.cli import main
from google_browser_scraper.pool import Pool, merge_stats
from google_browser_scraper.proxy import ProxyTemplate
from google_browser_scraper.scraper import ExitsRefused, Scraper, Settings


class Recording(FakeSession):
    def __init__(self, *args, worker, **kwargs):
        super().__init__(*args, **kwargs)
        self.worker = worker
        self.thread = threading.get_ident()

    def search(self, query):
        # Long enough that one worker cannot drain the queue before the other
        # starts; the tests below are about sharing it.
        time.sleep(0.01)
        return super().search(query)


def pool(scripts_per_worker, settings=None):
    """`scripts_per_worker[i]` is the list of identity scripts worker i mints."""
    FakeSession.opened = []
    FakeRelay.by_address = {}
    queues = [list(scripts) for scripts in scripts_per_worker]

    def make(i):
        def factory(proxy, **options):
            return Recording(proxy, script=queues[i].pop(0), worker=i, **options)

        return Scraper(
            ProxyTemplate("http://u-{session}:p@gate.example.com:7000"),
            settings or Settings(gap_seconds=(0, 0), dwell_seconds=(0, 0)),
            session_factory=factory,
            relay_factory=FakeRelay,
            resolver_factory=None,
            sleep=lambda seconds: None,
        )

    return Pool(make, len(scripts_per_worker), sleep=lambda seconds: None)


def test_every_query_is_answered_once_across_the_workers():
    p = pool([[[SERVED] * 10], [[SERVED] * 10]])
    queries = [f"q{i}" for i in range(8)]
    records = list(p.run(queries))
    assert sorted(r["search_parameters"]["q"] for r in records) == sorted(queries)
    assert all(r["search_metadata"]["status"] == "Success" for r in records)
    searched = [q for s in FakeSession.opened for q in s.searched]
    assert sorted(searched) == sorted(queries)
    assert p.not_run == []
    assert all(s.closed for s in FakeSession.opened)


def test_each_worker_keeps_its_browsers_on_its_own_thread_and_tags_its_records():
    p = pool([[[SERVED] * 10], [[SERVED] * 10]])
    records = list(p.run([f"q{i}" for i in range(8)]))
    threads = {s.worker: s.thread for s in FakeSession.opened}
    by_worker = {}
    for s in FakeSession.opened:
        by_worker.setdefault(s.worker, set()).update(s.searched)
    for r in records:
        assert r["search_parameters"]["q"] in by_worker[r["search_metadata"]["worker"]]
    assert len(set(threads.values())) == len(threads) == 2


def test_one_tripped_breaker_stops_the_pool_and_reports_what_was_not_run():
    settings = Settings(breaker=2, gap_seconds=(0, 0), dwell_seconds=(0, 0))
    p = pool([[[REFUSED], [REFUSED]], [[SERVED] * 50]], settings)
    queries = [f"q{i}" for i in range(40)]
    with pytest.raises(ExitsRefused):
        records = []
        for r in p.run(queries):
            records.append(r)
    answered = {r["search_parameters"]["q"] for r in records}
    assert answered.isdisjoint(p.not_run)
    assert len(answered) + len(p.not_run) <= len(queries)
    assert p.not_run, "the pool ran everything despite the breaker"
    assert all(s.closed for s in FakeSession.opened)


def test_pool_stats_add_up_the_workers():
    p = pool([[[SERVED] * 10], [[SERVED] * 10]])
    list(p.run([f"q{i}" for i in range(6)]))
    stats = p.stats
    assert stats.pages_served == 6
    assert stats.identities == 2
    assert stats.summary()["pages_served"] == 6


def test_merge_stats_of_nothing_is_empty():
    assert merge_stats([]).summary()["pages_served"] == 0


@pytest.mark.parametrize("workers", ["0", "17"])
def test_the_command_line_refuses_a_worker_count_out_of_range(workers, capsys):
    assert main(["search", "x", "--no-proxy", "--workers", workers]) == 1
    assert "workers must be between" in capsys.readouterr().err
