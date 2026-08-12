import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "examples/online_serving/duplexio/realtime_web"
APP_ROOT = ROOT / "app"
STATIC_ROOT = APP_ROOT / "static"

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_login_shows_logo_and_wordmark() -> None:
    login = (APP_ROOT / "login.html").read_text(encoding="utf-8")

    assert 'class="mark" src="/static/logo.png"' in login
    assert 'class="wordmark" src="/static/logo-wordmark.svg"' in login


def test_frontend_shows_role_grouped_transcript_and_collapsed_logs() -> None:
    index = (APP_ROOT / "index.html").read_text(encoding="utf-8")
    app = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert "<title>duplexio</title>" in index
    assert 'class="mark" src="/static/logo.png"' in index
    assert 'class="wordmark" src="/static/logo-wordmark.svg"' in index
    assert '<header class="header">' in index
    assert "justify-content: space-between;" in index
    assert "filter: brightness(0) invert(1);" in index
    assert "<video" not in index
    assert "Talk naturally" not in index
    assert '<section id="conversation"' in index
    assert '<details class="sampling-panel" open>' in index
    assert '<details class="logs-panel">' in index
    assert "activeMessage.role !== role" in app
    assert "response.audio_transcript.delta" in app
    assert "conversation.item.input_audio_transcription.delta" in app
    assert '<button id="record"' in index
    assert "microphone left, assistant right" in app


def test_recording_can_be_armed_before_session_start() -> None:
    index = (APP_ROOT / "index.html").read_text(encoding="utf-8")
    app = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert '<button id="record" type="button" aria-pressed="false">' in index
    assert '<button id="record" type="button" disabled' not in index
    socket_ready = app.index("await openSocket();")
    recording_start = app.index("if (recordingArmed) startRecording();", socket_ready)
    assert socket_ready < recording_start


def test_microphone_upload_waits_for_session_readiness() -> None:
    app = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert "if (message.type === 'session.updated' && !ready)" in app
    assert "const sendIntervalMs = 80;" in app
    assert "const playbackBufferMs = 160;" in app
    readiness = app.index("await openSocket();")
    running = app.index("running = true;", readiness)
    send_timer = app.index("window.setInterval(flushCapture", running)
    assert readiness < running < send_timer


def test_frontend_registers_tools_and_returns_function_outputs() -> None:
    index = (APP_ROOT / "index.html").read_text(encoding="utf-8")
    app = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert '<fieldset id="tool-picker"' in index
    assert "type: 'session.update'" in app
    assert "const sessionTools = enabledTools();" in app
    assert "tools: sessionTools" in app
    assert "tool_choice: sessionTools.length ? 'auto' : 'none'" in app
    assert "response.function_call_arguments.done" in app
    assert "type: 'function_call_output'" in app
    assert "Tool response → model" in app
    assert "`<tool_response>\\n${item.output}\\n</tool_response>`" in app
    assert "toolPicker.disabled = true" in app
    assert "toolPicker.disabled = false" in app


def test_frontend_includes_delayed_local_demo_tools() -> None:
    tools = json.loads((ROOT / "tools.json").read_text(encoding="utf-8"))
    app = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert [tool["function"]["name"] for tool in tools] == [
        "get_current_time",
        "get_weather",
        "search_web",
        "create_reminder",
    ]
    assert "await mockDelay(900);" in app
    assert "source: 'local demo'" in app


def test_frontend_can_select_a_model_voice() -> None:
    index = (APP_ROOT / "index.html").read_text(encoding="utf-8")
    app = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert '<select id="model-voice"' in index
    assert "voiceSelect.value = config.voice" in app
    assert "voice: voiceSelect.value" in app


def test_frontend_sends_sampling_parameters_before_session_start() -> None:
    index = (APP_ROOT / "index.html").read_text(encoding="utf-8")
    app = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert '<fieldset id="sampling-picker">' in index
    assert 'id="text-sampling-mode"' in index
    assert 'id="audio-temperature"' in index
    assert 'id="user-emit-temperature"' in index
    assert 'id="agent-emit-temperature"' in index
    assert 'id="tool-call-emit-temperature"' in index
    assert "const sessionSampling = samplingOptions();" in app
    assert "duplexio_sampling: sessionSampling" in app
    assert "emit.tool_call.toFixed(1)" in app
    assert "samplingPicker.disabled = true" in app
    assert "samplingPicker.disabled = false" in app


def test_recording_worklet_preserves_stereo_channels() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for the AudioWorklet regression test")

    script = textwrap.dedent(
        """
        const fs = require('fs');
        const vm = require('vm');

        const messages = [];
        global.sampleRate = 4;
        global.AudioWorkletProcessor = class {
          constructor() {
            this.port = {
              onmessage: null,
              postMessage: (message) => messages.push(message),
            };
          }
        };
        let Processor = null;
        global.registerProcessor = (_name, processor) => { Processor = processor; };
        vm.runInThisContext(fs.readFileSync(process.argv[1], 'utf8'));

        const processor = new Processor();
        processor.handle({ type: 'start' });
        processor.process(
          [[new Float32Array([1, 2]), new Float32Array([3, 4])]],
          [[new Float32Array(2), new Float32Array(2)]],
        );
        processor.handle({ type: 'stop' });
        const chunk = messages.find((message) => message.type === 'chunk');
        const pcm = Array.from(new Float32Array(chunk.pcm));
        if (pcm.join(',') !== '1,3,2,4') throw new Error(`unexpected stereo PCM: ${pcm}`);
        if (messages.at(-1).type !== 'stopped') throw new Error('missing stopped event');
        """
    )
    subprocess.run(
        [node, "-e", script, str(STATIC_ROOT / "recording_worklet.js")],
        check=True,
        capture_output=True,
        text=True,
    )


def test_playback_worklet_buffers_one_audio_frame() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for the AudioWorklet regression test")

    script = textwrap.dedent(
        """
        const fs = require('fs');
        const vm = require('vm');

        const messages = [];
        global.sampleRate = 1000;
        global.AudioWorkletProcessor = class {
          constructor() {
            this.port = {
              onmessage: null,
              postMessage: (message) => messages.push(message),
            };
          }
        };
        let Processor = null;
        global.registerProcessor = (_name, processor) => { Processor = processor; };
        vm.runInThisContext(fs.readFileSync(process.argv[1], 'utf8'));

        const processor = new Processor({ processorOptions: { playbackBufferMs: 80 } });
        const render = () => {
          const output = new Float32Array(40);
          processor.process([], [[output]]);
          return output;
        };
        const appendHalfFrame = () => processor.handle({
          type: 'audio',
          pcm: new Int16Array(40).fill(4096),
          responseId: 'response-1',
        });
        const assert = (condition, message) => {
          if (!condition) throw new Error(message);
        };

        appendHalfFrame();
        assert(render().every((sample) => sample === 0), 'playback started below one frame');
        appendHalfFrame();
        assert(render().every((sample) => sample === 0.125), 'playback did not start after one frame');

        while (processor.queuedFrames > 0) render();
        assert(!processor.playing, 'empty playback queue did not return to buffering');
        assert(messages.some((message) => message.type === 'buffering'), 'underrun was not reported');

        appendHalfFrame();
        assert(render().every((sample) => sample === 0), 'playback resumed below one frame');
        appendHalfFrame();
        assert(render().every((sample) => sample === 0.125), 'playback did not resume after one frame');

        processor.handle({ type: 'clear' });
        appendHalfFrame();
        processor.handle({ type: 'drain', responseId: 'response-1' });
        assert(render().every((sample) => sample === 0.125), 'drain did not flush a short response');
        """
    )
    subprocess.run(
        [node, "-e", script, str(STATIC_ROOT / "playback_worklet.js")],
        check=True,
        capture_output=True,
        text=True,
    )
