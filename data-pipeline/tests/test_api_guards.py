"""Balance guard and rate-limit handling for Stage A, with OpenRouter mocked (no network, no spend).
Run from data-pipeline/:  python -m unittest discover -s tests -t .
"""
import json
import unittest
from unittest import mock

import retry_failures as rf
import stage_a_generate as sa

GENS = [{"slug": "model/a"}, {"slug": "model/b"}]


def ok_call(sentence, slug, api_key, **kw):
    return {"source_text": sentence, "_generator_model": slug, "has_subtext": False}


def http_error(status, text="{}"):
    return mock.Mock(ok=False, status_code=status, text=text)


class BalanceGuardTests(unittest.TestCase):
    def test_remaining_balance_reads_credits_endpoint(self):
        resp = mock.Mock()
        resp.json.return_value = {"data": {"total_credits": 10.0, "total_usage": 9.4}}
        with mock.patch.object(sa.requests, "get", return_value=resp) as get:
            self.assertAlmostEqual(sa.remaining_balance("k"), 0.6)
            self.assertTrue(get.call_args.args[0].endswith("/api/v1/credits"))
            self.assertFalse(sa.balance_ok("k", 0.75, "test"))           # 0.60 < 0.75
            self.assertTrue(sa.balance_ok("k", 0.50, "test"))

    def test_unreadable_balance_does_not_stop_the_run(self):
        with mock.patch.object(sa.requests, "get", side_effect=sa.requests.ConnectionError("down")):
            self.assertIsNone(sa.remaining_balance("k"))
            self.assertTrue(sa.balance_ok("k", 0.75, "test"))

    def test_generate_stops_cleanly_and_records_unprocessed_seeds(self):
        seeds = [f"seed {i}" for i in range(60)]
        balances = iter([4.00, 0.70])                   # enough for chunk 1, under the guard for chunk 2
        with mock.patch.object(sa.requests, "get") as get, mock.patch.object(sa.time, "sleep"):
            get.return_value.json.side_effect = lambda: {"data": {"total_credits": 10.0,
                                                                  "total_usage": 10.0 - next(balances)}}
            results, failures, *_ = sa.generate(seeds, GENS, "k", sleep=0, call=ok_call, parallel=False)
        self.assertEqual(len(results), 25 * 2)                          # first chunk only
        self.assertEqual(len(failures), 35 * 2)                         # every unprocessed pair recorded
        self.assertEqual({f["source_text"] for f in failures}, set(seeds[25:]))
        self.assertTrue(all(f["error"].startswith("skipped -- balance guard") for f in failures))
        self.assertFalse(any(sa.is_model_failure(f) for f in failures))  # not held against any model

    def test_retry_pairs_stops_cleanly(self):
        pairs = [(f"seed {i}", "model/a") for i in range(30)]
        checks = iter([True, False])
        with mock.patch.object(sa.time, "sleep"):
            results, still = rf.retry_pairs(pairs, "k", sleep=0, call=ok_call, check_balance=lambda _: next(checks))
        self.assertEqual(len(results), 25)
        self.assertEqual([s for s, _ in pairs[25:]], [r["source_text"] for r in still])

    def test_http_402_stops_the_run_instead_of_failing_every_call(self):
        calls = []

        def broke_after_3(sentence, slug, api_key, **kw):
            calls.append(slug)
            if len(calls) > 3:
                raise sa.OutOfCredits("HTTP 402")
            return ok_call(sentence, slug, api_key)
        results, failures, _, consecutive, tripped = sa.generate(
            ["s0", "s1", "s2"], GENS, "k", sleep=0, call=broke_after_3, check_balance=lambda _: True,
            parallel=False)
        self.assertEqual(len(calls), 4)                                  # no calls after the 402
        self.assertEqual(len(results), 3)
        self.assertEqual(len(failures), 3)                               # s1/model b, s2/a, s2/b
        self.assertFalse(tripped)
        self.assertFalse(any(sa.is_model_failure(f) for f in failures))


class RateLimitTests(unittest.TestCase):
    def test_429_backs_off_then_succeeds(self):
        responses = [http_error(429), http_error(429)]
        with mock.patch.object(sa, "call_model", side_effect=lambda *a, **k: _raise_or_ok(responses, a)), \
                mock.patch.object(sa.time, "sleep") as sleep:
            row = sa.call_model_with_retry("s", "model/a", "k")
        self.assertEqual(row["_generator_model"], "model/a")
        self.assertEqual([c.args[0] for c in sleep.call_args_list], list(sa.RATE_LIMIT_WAITS[:2]))

    def test_persistent_429_raises_rate_limited(self):
        with mock.patch.object(sa, "call_model", side_effect=RuntimeError("HTTP 429 from OpenRouter")), \
                mock.patch.object(sa.time, "sleep") as sleep:
            with self.assertRaises(sa.RateLimited):
                sa.call_model_with_retry("s", "model/a", "k")
        self.assertEqual(sleep.call_count, len(sa.RATE_LIMIT_WAITS))

    def test_429s_never_trip_the_breaker_or_count_as_failures(self):
        def a_is_rate_limited(sentence, slug, api_key, **kw):
            if slug == "model/a":
                raise sa.RateLimited("still 429")
            return ok_call(sentence, slug, api_key)
        seeds = [f"seed {i}" for i in range(10)]                         # 10 > breaker threshold of 4
        results, failures, _, consecutive, tripped = sa.generate(
            seeds, GENS, "k", sleep=0, call=a_is_rate_limited, check_balance=lambda _: True, parallel=False)
        self.assertEqual(tripped, set())
        self.assertEqual(consecutive["model/a"], 0)
        self.assertEqual(len(failures), 10)                              # kept for a later retry...
        self.assertFalse(any(sa.is_model_failure(f) for f in failures))  # ...but not model failures
        self.assertEqual(len(results), 10)                               # model b unaffected

    def test_real_errors_still_trip_the_breaker(self):
        def a_broken(sentence, slug, api_key, **kw):
            if slug == "model/a":
                raise RuntimeError("HTTP 500 from OpenRouter")
            return ok_call(sentence, slug, api_key)
        _, failures, _, _, tripped = sa.generate([f"s{i}" for i in range(6)], GENS, "k", sleep=0,
                                                 call=a_broken, check_balance=lambda _: True, parallel=False)
        self.assertEqual(tripped, {"model/a"})
        self.assertEqual(sum(sa.is_model_failure(f) for f in failures), 4)   # 4 real, then skipped


class SourceTextTests(unittest.TestCase):
    def test_rows_are_keyed_by_the_seed_not_the_models_echo(self):
        seed = "I absolutely love spending my Saturday fixing someone else’s mess."
        body = {"source_lang": "en", "source_text": seed.replace("’", "'"), "translation": "Обожаю...",
                "has_subtext": True, "category": "sarcasm", "nuance_note": "Sarcasm."}
        resp = mock.Mock(ok=True, status_code=200)
        resp.json.return_value = {"choices": [{"message": {"content": json.dumps(body)}}]}
        with mock.patch.object(sa.requests, "post", return_value=resp):
            row = sa.call_model(seed, "model/a", "k")
        self.assertEqual(row["source_text"], seed)
        self.assertEqual(row["_model_echoed_source_text"], seed.replace("’", "'"))

    def test_rekey_existing_rows_to_their_seed(self):
        seeds = ["I’m so grateful you took the last piece of cake without asking.", "Проходите — доктор ждёт…"]
        rows = [{"source_text": "I'm so grateful you took the last piece of cake without asking."},
                {"source_text": "Проходите - доктор ждёт..."},
                {"source_text": seeds[0]},
                {"source_text": "Something no seed matches."}]
        self.assertEqual(sa.rekey_to_seeds(rows, seeds), 2)
        self.assertEqual([r["source_text"] for r in rows[:3]], [seeds[0], seeds[1], seeds[0]])
        self.assertEqual(rows[3]["source_text"], "Something no seed matches.")

    def test_rekey_hyphens_typo_fixes_and_partial_echoes(self):
        seeds = ["Молодой человек, вы что‑то потеряли.",                        # non-breaking hyphen
                 "Как заботливо с твоей стороны забыть мой ден рождения.",           # typo in the seed
                 "Спасибо за вашу помощь, доктор. — Спасибо, что выручил.",          # two-speaker seed
                 "Вы не подскажете, где здесь аптека?",
                 "Вы не подскажете, где здесь ближайшая аптека?"]                    # near-duplicate seed
        rows = [{"source_text": "Молодой человек, вы что-то потеряли."},
                {"source_text": "Как заботливо с твоей стороны забыть мой день рождения."},
                {"source_text": "Спасибо за вашу помощь, доктор."},                  # partial echo: leave
                {"source_text": "Вы не подскажете, где здесь аптека ?"}]               # ambiguous: leave
        self.assertEqual(sa.rekey_to_seeds(rows, seeds), 2)
        self.assertEqual([r["source_text"] for r in rows], [seeds[0], seeds[1], "Спасибо за вашу помощь, доктор.",
                                                            "Вы не подскажете, где здесь аптека ?"])


class ParallelResumeTests(unittest.TestCase):
    def test_generators_run_concurrently_per_seed(self):
        import threading
        import time as _time
        gens = [{"slug": f"model/{c}", "provider": {"order": [c]}} for c in "abcd"]
        live, peak, lock, routed = [0], [0], threading.Lock(), []

        def slow_call(sentence, slug, api_key, provider=None, api_model=None):
            with lock:
                live[0] += 1
                peak[0] = max(peak[0], live[0])
                routed.append((slug, provider))
            _time.sleep(0.05)
            with lock:
                live[0] -= 1
            return ok_call(sentence, slug, api_key)
        written = []
        results, failures, *_ = sa.generate([f"s{i}" for i in range(3)], gens, "k", sleep=0, call=slow_call,
                                            check_balance=lambda _: True, on_result=written.append)
        self.assertEqual(peak[0], 4)                                      # all 4 models at once
        self.assertEqual(len(results), 12)
        self.assertEqual(written, results)                                # every row streamed out
        self.assertEqual([r["_generator_model"] for r in results[:4]], [g["slug"] for g in gens])  # roster order
        self.assertIn(("model/c", {"order": ["c"]}), routed)              # provider routing passed through
        self.assertFalse(failures)

    def test_resume_skips_pairs_already_done(self):
        calls = []

        def rec(sentence, slug, api_key, **kw):
            calls.append((sentence, slug))
            return ok_call(sentence, slug, api_key)
        done = {("s0", "model/a"), ("s0", "model/b"), ("s1", "model/a")}
        results, *_ = sa.generate(["s0", "s1"], GENS, "k", sleep=0, call=rec, check_balance=lambda _: True,
                                  done=done)
        self.assertEqual(calls, [("s1", "model/b")])
        self.assertEqual(len(results), 1)

    def test_retry_uses_roster_routing(self):
        seen = []

        def rec(sentence, slug, api_key, provider=None, api_model=None):
            seen.append((slug, provider, api_model))
            return ok_call(sentence, slug, api_key)
        routing = {"model/a": {"slug": "model/a", "provider": {"order": ["x"]}},
                   "model/b": {"slug": "model/b", "api_model": "model/b:floor"}}
        rf.retry_pairs([("s", "model/a"), ("s", "model/b")], "k", sleep=0, call=rec,
                       check_balance=lambda _: True, routing=routing)
        self.assertEqual(seen, [("model/a", {"order": ["x"]}, None), ("model/b", None, "model/b:floor")])


def _raise_or_ok(responses, args):
    if responses:
        responses.pop(0)
        raise RuntimeError("HTTP 429 from OpenRouter for model=model/a: slow down")
    return ok_call(args[0], args[1], args[2])


if __name__ == "__main__":
    unittest.main()
