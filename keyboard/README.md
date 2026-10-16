# keyboard/

Android side of Podtekst. Today it holds the on-device engine and a test app; the FlorisBoard-based
keyboard will be added here and reuse the same engine.

| Module | What it does |
| --- | --- |
| `engine` | Loads a `.litertlm` model with [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM), builds the exact training prompt (`PodtekstPrompt`, checked against `model/podtekst_sft/prompt.py` by a unit test), parses the JSON reply (`ReplyParser`, a port of `parse_reply()`), lists and imports model files (`ModelStore`). |
| `lab` | Podtekst Lab, a test app: pick a model and CPU/GPU, try single messages with timings, run a whole test file and save the results for `model/score_device.py`. |

Generation matches the evaluation: greedy decoding, thinking off, at most 256 tokens, a new conversation
per message. The app passes its own ChatML template because the base model's template expects plain
strings and LiteRT-LM passes message content as a list of parts.

## Build and install

Needs Android Studio (for its JDK and the Android SDK). From this folder:

```powershell
$env:JAVA_HOME = "<Android Studio>\jbr"
.\gradlew.bat :engine:testDebugUnitTest :lab:assembleDebug
adb install -r lab\build\outputs\apk\debug\lab-debug.apk
```

## Models and test files

The app reads from its private storage. Open it once, then copy files in through `/data/local/tmp`
(files adb writes into `/sdcard/Android/data/<app>` are not readable by the app):

```powershell
adb push model.litertlm /data/local/tmp/
adb shell chmod 644 /data/local/tmp/model.litertlm
adb shell run-as com.podtekst.lab cp /data/local/tmp/model.litertlm files/models/
adb shell rm /data/local/tmp/model.litertlm
```

Test files (`.jsonl`, dataset rows or `sft_test.jsonl` items) go to `files/probes/` the same way. Results
are written to `files/results/<model>__<probe>__<time>.jsonl`; copy them out with
`adb exec-out run-as com.podtekst.lab cat files/results/<name>` (redirect with `cmd /c "... > file"`
on Windows so the bytes are not re-encoded) and score them with `python model/score_device.py`.
