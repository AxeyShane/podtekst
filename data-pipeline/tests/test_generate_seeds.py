"""generate_seeds.py: routing reaches the request, empty replies are retried (OpenRouter mocked)."""
import unittest
from unittest import mock

import generate_seeds as gs


def reply(content):
    r = mock.Mock(ok=True, status_code=200)
    r.json.return_value = {"choices": [{"message": {"content": content}}]}
    return r


class GenerateSeedsTests(unittest.TestCase):
    def test_routing_and_reasoning_reach_the_request(self):
        with mock.patch.object(gs.requests, "post", return_value=reply("1. Ты где был?\n2. Как дела?")) as post:
            lines = gs.call_with_retry("idiom", "ru", 2, "deepseek/deepseek-v4-flash", "k",
                                       provider={"order": ["streamlake"]}, api_model=None)
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["provider"], {"order": ["streamlake"]})
        self.assertEqual(body["reasoning"], {"effort": "low"})
        self.assertEqual(lines, ["Ты где был?", "Как дела?"])

    def test_empty_content_is_retried(self):
        with mock.patch.object(gs.requests, "post", side_effect=[reply(None), reply("Ну ты даёшь!")]), \
                mock.patch.object(gs.time, "sleep"):
            self.assertEqual(gs.call_with_retry("idiom", "ru", 1, "m", "k"), ["Ну ты даёшь!"])


if __name__ == "__main__":
    unittest.main()
