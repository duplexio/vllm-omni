(() => {
  'use strict';

  const config = window.DUPLEXIO_CONFIG || {};
  const inputRate = Number(config.inputSampleRate || 24000);
  const outputRate = 24000;
  const playbackBufferMs = 80;
  const sendIntervalMs = 80;
  const tools = Array.isArray(config.tools) ? config.tools : [];

  const startButton = document.getElementById('start');
  const muteButton = document.getElementById('mute');
  const recordButton = document.getElementById('record');
  const voiceSelect = document.getElementById('model-voice');
  const samplingPicker = document.getElementById('sampling-picker');
  const textSamplingMode = document.getElementById('text-sampling-mode');
  const textTemperature = document.getElementById('text-temperature');
  const textTopK = document.getElementById('text-top-k');
  const textTopP = document.getElementById('text-top-p');
  const audioTemperature = document.getElementById('audio-temperature');
  const audioTopK = document.getElementById('audio-top-k');
  const userEmitTemperature = document.getElementById('user-emit-temperature');
  const agentEmitTemperature = document.getElementById('agent-emit-temperature');
  const toolCallEmitTemperature = document.getElementById('tool-call-emit-temperature');
  const samplingSeed = document.getElementById('sampling-seed');
  const toolPicker = document.getElementById('tool-picker');
  const statusElement = document.getElementById('status');
  const statusLabel = document.getElementById('status-label');
  const detailElement = document.getElementById('detail');
  const meterFill = document.getElementById('meter-fill');
  const conversationElement = document.getElementById('conversation');
  const logElement = document.getElementById('log');

  let socket = null;
  let mediaStream = null;
  let captureContext = null;
  let captureNode = null;
  let playbackContext = null;
  let playbackNode = null;
  let recordingMicSource = null;
  let recordingMerger = null;
  let recordingNode = null;
  let recordingSink = null;
  let captureRate = inputRate;
  let playbackRate = outputRate;
  let captureChunks = [];
  let sendTimer = null;
  let running = false;
  let muted = false;
  let responseId = null;
  let activeMessage = null;
  const toolHandlers = new Map();
  const toolCards = new Map();

  function mockDelay(milliseconds) {
    return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
  }

  function readableToolName(name) {
    return name.replaceAll('_', ' ').replace(/^./, (letter) => letter.toUpperCase());
  }

  async function getCurrentTime({ time_zone: timeZone } = {}) {
    await mockDelay(600);
    const options = {
      dateStyle: 'full',
      timeStyle: 'long',
      ...(timeZone ? { timeZone } : {}),
    };
    return {
      iso: new Date().toISOString(),
      formatted: new Intl.DateTimeFormat(undefined, options).format(new Date()),
      time_zone: timeZone || Intl.DateTimeFormat().resolvedOptions().timeZone,
      source: 'local demo',
    };
  }

  async function getWeather({ location, units = 'celsius' }) {
    await mockDelay(900);
    const seed = [...location].reduce((total, character) => total + character.codePointAt(0), 0);
    const conditions = ['Clear skies', 'Partly cloudy', 'Light rain', 'Overcast'];
    const celsius = 8 + seed % 19;
    const temperature = units === 'fahrenheit' ? Math.round(celsius * 9 / 5 + 32) : celsius;
    return {
      location,
      condition: conditions[seed % conditions.length],
      temperature,
      units,
      humidity_percent: 45 + seed % 40,
      wind_kph: 5 + seed % 24,
      observed_at: new Date().toISOString(),
      source: 'local demo',
    };
  }

  async function searchWeb({ query, max_results: maxResults = 3 }) {
    await mockDelay(1100);
    const count = Math.min(maxResults, 3);
    return {
      query,
      results: Array.from({ length: count }, (_value, index) => ({
        title: `${query} — result ${index + 1}`,
        url: `https://example.com/search/${index + 1}`,
        snippet: `A concise demonstration result related to “${query}”.`,
      })),
      source: 'local demo',
    };
  }

  async function createReminder({ title, delay_minutes: delayMinutes }) {
    await mockDelay(750);
    return {
      id: `reminder_${crypto.randomUUID()}`,
      title,
      status: 'scheduled',
      scheduled_for: new Date(Date.now() + delayMinutes * 60_000).toISOString(),
      source: 'local demo',
    };
  }

  toolHandlers.set('get_current_time', getCurrentTime);
  toolHandlers.set('get_weather', getWeather);
  toolHandlers.set('search_web', searchWeb);
  toolHandlers.set('create_reminder', createReminder);
  let recordingArmed = false;
  let recording = false;
  let recordingChunks = [];
  let recordingFrames = 0;
  let recordingStopped = null;

  function assetUrl(path) {
    const version = String(config.appVersion || '').trim();
    return version ? `${path}?v=${encodeURIComponent(version)}` : path;
  }

  function setStatus(label, kind = '') {
    statusLabel.textContent = label;
    statusElement.className = `status ${kind}`;
  }

  function populateVoices() {
    const voices = Array.isArray(config.voices) ? config.voices : [];
    const options = voices.map((voice) => new Option(voice.label, voice.id));
    voiceSelect.replaceChildren(...options);
    voiceSelect.value = config.voice;
  }

  function populateTools() {
    const options = tools.map((tool) => {
      const name = tool.function.name;
      const label = document.createElement('label');
      label.className = 'tool-option';
      const toggle = document.createElement('input');
      toggle.className = 'tool-toggle';
      toggle.type = 'checkbox';
      toggle.value = name;
      toggle.checked = true;
      const text = document.createElement('span');
      const title = document.createElement('span');
      title.className = 'tool-option-name';
      title.textContent = readableToolName(name);
      const description = document.createElement('span');
      description.className = 'tool-option-description';
      description.textContent = tool.function.description || '';
      text.append(title, description);
      label.append(toggle, text);
      return label;
    });
    toolPicker.replaceChildren(...options);
  }

  function populateSampling() {
    const sampling = config.sampling || {};
    const text = sampling.text || {};
    const audio = sampling.audio || {};
    const emit = sampling.emit || {};
    textSamplingMode.value = text.mode || '';
    textTemperature.value = text.temperature ?? '';
    textTopK.value = text.top_k ?? '';
    textTopP.value = text.top_p ?? '';
    audioTemperature.value = audio.temperature ?? '';
    audioTopK.value = audio.top_k ?? '';
    userEmitTemperature.value = emit.user ?? '';
    agentEmitTemperature.value = emit.agent ?? '';
    toolCallEmitTemperature.value = Number.isInteger(emit.tool_call)
      ? emit.tool_call.toFixed(1)
      : emit.tool_call ?? '';
  }

  function samplingNumber(input) {
    if (input.value === '') return null;
    if (!input.reportValidity()) {
      throw new Error(`Invalid sampling parameter: ${input.id}`);
    }
    return input.valueAsNumber;
  }

  function samplingOptions() {
    const text = {};
    const audio = {};
    const emit = {};
    if (textSamplingMode.value) text.mode = textSamplingMode.value;
    const textValues = [
      ['temperature', textTemperature],
      ['top_k', textTopK],
      ['top_p', textTopP],
    ];
    const audioValues = [
      ['temperature', audioTemperature],
      ['top_k', audioTopK],
    ];
    const emitValues = [
      ['user', userEmitTemperature],
      ['agent', agentEmitTemperature],
      ['tool_call', toolCallEmitTemperature],
    ];
    for (const [name, input] of textValues) {
      const value = samplingNumber(input);
      if (value !== null) text[name] = value;
    }
    for (const [name, input] of audioValues) {
      const value = samplingNumber(input);
      if (value !== null) audio[name] = value;
    }
    for (const [name, input] of emitValues) {
      const value = samplingNumber(input);
      if (value !== null) emit[name] = value;
    }
    const seed = samplingNumber(samplingSeed);
    return {
      ...(seed === null ? {} : { seed }),
      ...(Object.keys(text).length ? { text } : {}),
      ...(Object.keys(audio).length ? { audio } : {}),
      ...(Object.keys(emit).length ? { emit } : {}),
    };
  }

  function enabledTools() {
    const enabledNames = new Set(
      [...toolPicker.querySelectorAll('.tool-toggle:checked')]
        .map((toggle) => toggle.value),
    );
    return tools.filter((tool) => enabledNames.has(tool.function.name));
  }

  function log(message) {
    const line = `${new Date().toLocaleTimeString([], { hour12: false })}  ${message}`;
    logElement.textContent = `${logElement.textContent}${line}\n`.slice(-8000);
    logElement.scrollTop = logElement.scrollHeight;
  }

  function resetConversation() {
    conversationElement.replaceChildren();
    const emptyMessage = document.createElement('p');
    emptyMessage.id = 'empty-message';
    emptyMessage.className = 'empty-message';
    emptyMessage.textContent = 'Transcribed model tokens will appear here.';
    conversationElement.appendChild(emptyMessage);
    activeMessage = null;
    toolCards.clear();
  }

  function appendTranscript(role, delta) {
    if (!delta) return;
    if (!activeMessage || activeMessage.role !== role) {
      const emptyMessage = document.getElementById('empty-message');
      if (emptyMessage) emptyMessage.remove();
      const message = document.createElement('div');
      message.className = `message message-${role}`;
      const label = document.createElement('div');
      label.className = 'message-role';
      label.textContent = role === 'user' ? 'You' : 'Assistant';
      const text = document.createElement('div');
      text.className = 'message-text';
      message.append(label, text);
      conversationElement.appendChild(message);
      activeMessage = { role, text, value: '' };
    }
    activeMessage.value += delta;
    activeMessage.text.textContent = activeMessage.value;
    conversationElement.scrollTop = conversationElement.scrollHeight;
  }

  function sendToolResult(callId, output) {
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      throw new Error('The Realtime session is not connected.');
    }
    const value = typeof output === 'string' ? output : JSON.stringify(output);
    socket.send(JSON.stringify({
      type: 'conversation.item.create',
      item: {
        id: `item_tool_output_${crypto.randomUUID()}`,
        type: 'function_call_output',
        status: 'completed',
        call_id: callId,
        output: value,
      },
    }));
    const card = toolCards.get(callId);
    if (card) {
      card.status.textContent = 'Injecting result';
      card.input.value = value;
      card.input.disabled = true;
      card.button.disabled = true;
    }
    log(`tool result sent: ${callId}`);
  }

  function appendToolResponse(item) {
    const emptyMessage = document.getElementById('empty-message');
    if (emptyMessage) emptyMessage.remove();
    activeMessage = null;

    const message = document.createElement('div');
    message.className = 'message message-tool';
    const label = document.createElement('div');
    label.className = 'message-role';
    label.textContent = 'Tool response → model';
    const body = document.createElement('pre');
    body.className = 'tool-arguments';
    body.textContent = `<tool_response>\n${item.output}\n</tool_response>`;
    message.append(label, body);
    conversationElement.appendChild(message);
    conversationElement.scrollTop = conversationElement.scrollHeight;

    const card = toolCards.get(item.call_id);
    if (card) card.status.textContent = 'Injected into model stream';
  }

  function appendToolCall(call) {
    const emptyMessage = document.getElementById('empty-message');
    if (emptyMessage) emptyMessage.remove();
    activeMessage = null;

    const message = document.createElement('div');
    message.className = 'message message-tool';
    const label = document.createElement('div');
    label.className = 'message-role';
    label.textContent = 'Tool call';
    const body = document.createElement('div');
    body.className = 'tool-call';
    const name = document.createElement('strong');
    name.textContent = call.name;
    const argumentsElement = document.createElement('pre');
    argumentsElement.className = 'tool-arguments';
    argumentsElement.textContent = JSON.stringify(call.arguments, null, 2);
    const status = document.createElement('div');
    status.className = 'tool-status';
    status.textContent = `Call ${call.id}`;
    const input = document.createElement('textarea');
    input.className = 'tool-result';
    input.rows = 3;
    input.placeholder = 'Tool result';
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = 'Send result';
    button.addEventListener('click', () => sendToolResult(call.id, input.value));
    body.append(name, argumentsElement, status, input, button);
    message.append(label, body);
    conversationElement.appendChild(message);
    conversationElement.scrollTop = conversationElement.scrollHeight;
    toolCards.set(call.id, { status, input, button });

    const handler = toolHandlers.get(call.name);
    if (!handler) return;
    status.textContent = 'Running locally';
    Promise.resolve()
      .then(() => handler(call.arguments))
      .then((result) => sendToolResult(call.id, result))
      .catch((error) => sendToolResult(call.id, { error: error.message || String(error) }));
  }

  window.duplexioRegisterTool = (name, handler) => toolHandlers.set(name, handler);
  window.duplexioSendToolResult = sendToolResult;

  function base64FromBytes(bytes) {
    let binary = '';
    const chunkSize = 0x8000;
    for (let offset = 0; offset < bytes.length; offset += chunkSize) {
      binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
    }
    return btoa(binary);
  }

  function bytesFromBase64(encoded) {
    const binary = atob(encoded);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    return bytes;
  }

  function resampleInt16(input, sourceRate, targetRate) {
    if (sourceRate === targetRate) return input;
    const outputLength = Math.max(1, Math.round(input.length * targetRate / sourceRate));
    const output = new Int16Array(outputLength);
    for (let index = 0; index < output.length; index += 1) {
      const position = index * sourceRate / targetRate;
      const left = Math.floor(position);
      const right = Math.min(left + 1, input.length - 1);
      const fraction = position - left;
      output[index] = input[left] + (input[right] - input[left]) * fraction;
    }
    return output;
  }

  function int16ToFloat32Base64(pcm) {
    const bytes = new Uint8Array(pcm.length * 4);
    const view = new DataView(bytes.buffer);
    for (let index = 0; index < pcm.length; index += 1) {
      view.setFloat32(index * 4, pcm[index] / 32768, true);
    }
    return base64FromBytes(bytes);
  }

  function float32BytesToInt16(bytes) {
    const count = Math.floor(bytes.byteLength / 4);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const pcm = new Int16Array(count);
    for (let index = 0; index < count; index += 1) {
      const sample = Math.max(-1, Math.min(1, view.getFloat32(index * 4, true)));
      pcm[index] = sample < 0 ? sample * 32768 : sample * 32767;
    }
    return pcm;
  }

  function int16BytesToInt16(bytes) {
    const count = Math.floor(bytes.byteLength / 2);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const pcm = new Int16Array(count);
    for (let index = 0; index < count; index += 1) {
      pcm[index] = view.getInt16(index * 2, true);
    }
    return pcm;
  }

  function writeAscii(view, offset, value) {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index));
    }
  }

  function stereoWavBlob(chunks, frameCount, sampleRate) {
    const channelCount = 2;
    const bytesPerSample = 2;
    const blockAlign = channelCount * bytesPerSample;
    const buffer = new ArrayBuffer(44 + frameCount * blockAlign);
    const view = new DataView(buffer);
    writeAscii(view, 0, 'RIFF');
    view.setUint32(4, buffer.byteLength - 8, true);
    writeAscii(view, 8, 'WAVE');
    writeAscii(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, channelCount, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bytesPerSample * 8, true);
    writeAscii(view, 36, 'data');
    view.setUint32(40, frameCount * blockAlign, true);

    let byteOffset = 44;
    for (const chunk of chunks) {
      for (let index = 0; index < chunk.length; index += 1) {
        const sample = Math.max(-1, Math.min(1, chunk[index]));
        const pcm = sample < 0 ? sample * 32768 : sample * 32767;
        view.setInt16(byteOffset, pcm, true);
        byteOffset += bytesPerSample;
      }
    }
    return new Blob([buffer], { type: 'audio/wav' });
  }

  function downloadRecording(sampleRate) {
    if (recordingFrames === 0) {
      log('recording stopped without audio');
      return;
    }
    const blob = stereoWavBlob(recordingChunks, recordingFrames, sampleRate);
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    link.href = url;
    link.download = `duplexio-${timestamp}.wav`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    const seconds = recordingFrames / sampleRate;
    log(`recording downloaded (${seconds.toFixed(1)} s; microphone left, assistant right)`);
  }

  function updateRecordingButton() {
    recordButton.textContent = recording
      ? 'Stop & download'
      : recordingArmed
        ? 'Recording enabled'
        : 'Record session';
    recordButton.classList.toggle('armed', recordingArmed && !recording);
    recordButton.classList.toggle('recording', recording);
    recordButton.setAttribute('aria-pressed', recordingArmed ? 'true' : 'false');
  }

  function startRecording() {
    if (!recordingNode || recording) return;
    recordingChunks = [];
    recordingFrames = 0;
    recording = true;
    recordingNode.port.postMessage({ type: 'start' });
    updateRecordingButton();
    log('recording started (microphone left, assistant right)');
  }

  async function stopRecording() {
    if (!recordingNode || !recording) return;
    recording = false;
    const stopped = new Promise((resolve) => { recordingStopped = resolve; });
    recordingNode.port.postMessage({ type: 'stop' });
    await stopped;
    recordingStopped = null;
    updateRecordingButton();
  }

  function mergeCaptureChunks() {
    const length = captureChunks.reduce((total, chunk) => total + chunk.length, 0);
    const merged = new Int16Array(length);
    let offset = 0;
    for (const chunk of captureChunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    captureChunks = [];
    return merged;
  }

  function updateMeter(pcm) {
    let peak = 0;
    for (let index = 0; index < pcm.length; index += 8) {
      peak = Math.max(peak, Math.abs(pcm[index]));
    }
    meterFill.style.width = `${Math.min(100, peak / 32768 * 150)}%`;
  }

  function realtimeUrl() {
    const url = new URL(config.realtimePath, window.location.href);
    url.protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    url.searchParams.set('duplex', '1');
    url.searchParams.set('model', config.model);
    url.searchParams.set('autostart', '0');
    return url.toString();
  }

  function flushCapture() {
    if (!running || muted || !socket || socket.readyState !== WebSocket.OPEN) return;
    if (captureChunks.length === 0) return;
    const pcm = resampleInt16(mergeCaptureChunks(), captureRate, inputRate);
    socket.send(JSON.stringify({
      type: 'input_audio_buffer.append',
      audio: int16ToFloat32Base64(pcm),
      format: 'pcm_f32le',
      sample_rate_hz: inputRate,
    }));
  }

  async function openAudio() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Microphone access requires HTTPS; use the Tailscale URL.');
    }
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) throw new Error('This browser does not support Web Audio.');

    playbackContext = new AudioContextClass({ sampleRate: outputRate });
    playbackRate = playbackContext.sampleRate;
    await Promise.all([
      playbackContext.audioWorklet.addModule(assetUrl('static/playback_worklet.js')),
      playbackContext.audioWorklet.addModule(assetUrl('static/recording_worklet.js')),
    ]);
    playbackNode = new AudioWorkletNode(playbackContext, 'duplexio-playback', {
      processorOptions: { playbackBufferMs },
    });
    playbackNode.port.onmessage = (event) => {
      const message = event.data || {};
      if (message.type === 'started') detailElement.textContent = 'Speaking';
      if (message.type === 'buffering') {
        detailElement.textContent = 'Buffering audio';
        log(`playback underrun ${message.underruns}; buffering ${playbackBufferMs} ms`);
      }
      if (message.type === 'drained') {
        if (socket && socket.readyState === WebSocket.OPEN && message.responseId) {
          socket.send(JSON.stringify({
            type: 'playback.ack',
            response_id: message.responseId,
            item_id: `item_${message.responseId}`,
            played_ms: message.playedMs,
            committed_ms: message.playedMs,
          }));
        }
        if (running) detailElement.textContent = 'Listening';
      }
    };
    playbackNode.connect(playbackContext.destination);
    await playbackContext.resume();

    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        sampleRate: { ideal: inputRate },
      },
    });
    try {
      captureContext = new AudioContextClass({ sampleRate: inputRate });
    } catch (_error) {
      captureContext = new AudioContextClass();
    }
    captureRate = captureContext.sampleRate;
    await captureContext.audioWorklet.addModule(assetUrl('static/capture_worklet.js'));
    const source = captureContext.createMediaStreamSource(mediaStream);
    captureNode = new AudioWorkletNode(captureContext, 'duplexio-capture');
    captureNode.port.onmessage = (event) => {
      const pcm = new Int16Array(event.data);
      updateMeter(pcm);
      if (running && !muted) captureChunks.push(pcm);
    };
    const silentSink = captureContext.createGain();
    silentSink.gain.value = 0;
    source.connect(captureNode);
    captureNode.connect(silentSink).connect(captureContext.destination);
    await captureContext.resume();

    recordingMicSource = playbackContext.createMediaStreamSource(mediaStream);
    recordingMerger = playbackContext.createChannelMerger(2);
    recordingMicSource.connect(recordingMerger, 0, 0);
    playbackNode.connect(recordingMerger, 0, 1);
    recordingNode = new AudioWorkletNode(playbackContext, 'duplexio-recorder', {
      channelCount: 2,
      channelCountMode: 'explicit',
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [2],
    });
    recordingSink = playbackContext.createGain();
    recordingSink.gain.value = 0;
    recordingMerger.connect(recordingNode).connect(recordingSink).connect(playbackContext.destination);
    recordingNode.port.onmessage = (event) => {
      const message = event.data || {};
      if (message.type === 'chunk' && message.pcm) {
        const chunk = new Float32Array(message.pcm);
        recordingChunks.push(chunk);
        recordingFrames += chunk.length / 2;
      }
      if (message.type === 'stopped') {
        downloadRecording(Number(message.sampleRate || playbackRate));
        if (recordingStopped) recordingStopped();
      }
    };
  }

  function openSocket() {
    return new Promise((resolve, reject) => {
      const url = realtimeUrl();
      const sessionTools = enabledTools();
      const sessionSampling = samplingOptions();
      socket = new WebSocket(url);
      let ready = false;
      socket.onopen = () => {
        socket.send(JSON.stringify({
          type: 'session.update',
          session: {
            model: config.model,
            modalities: ['audio', 'text'],
            voice: voiceSelect.value,
            response_format: 'pcm',
            tools: sessionTools,
            tool_choice: sessionTools.length ? 'auto' : 'none',
            extra_body: {
              full_duplex: true,
              auto_response: true,
              start_role: 'agent',
              duplexio_sampling: sessionSampling,
            },
          },
        }));
        detailElement.textContent = 'Preparing model';
        log('connected; preparing session');
      };
      socket.onmessage = (event) => {
        if (typeof event.data !== 'string') return;
        let message;
        try {
          message = JSON.parse(event.data);
        } catch (error) {
          log(`invalid server message: ${error.message}`);
          return;
        }
        handleServerEvent(message);
        if (message.type === 'session.updated' && !ready) {
          ready = true;
          log('session ready');
          resolve();
        } else if (message.type === 'error' && !ready) {
          reject(new Error(JSON.stringify(message.error || message)));
        }
      };
      socket.onerror = () => {
        if (!ready) reject(new Error('WebSocket connection failed.'));
      };
      socket.onclose = (event) => {
        log(`disconnected (${event.code})`);
        if (!ready) reject(new Error(`WebSocket closed before session readiness (${event.code}).`));
        if (running) stopSession();
      };
    });
  }

  function handleServerEvent(event) {
    if (event.type === 'session.created') {
      setStatus('Preparing');
      return;
    }
    if (event.type === 'session.updated') {
      setStatus('Connected', 'online');
      return;
    }
    if (event.type === 'response.created' || event.type === 'response.speak') {
      responseId = event.response_id || (event.response && event.response.id) || responseId;
      detailElement.textContent = 'Speaking';
      return;
    }
    if (event.type === 'response.audio.delta') {
      const encoded = event.delta || event.audio || (event.response && event.response.audio);
      if (!encoded || !playbackNode) return;
      responseId = event.response_id || responseId;
      const bytes = bytesFromBase64(encoded);
      const format = String(event.format || event.audio_format || '').toLowerCase();
      const sourceRate = Number(event.sample_rate_hz || event.sample_rate || outputRate);
      const pcm = format.includes('f32')
        ? float32BytesToInt16(bytes)
        : int16BytesToInt16(bytes);
      const playbackPcm = resampleInt16(pcm, sourceRate, playbackRate);
      playbackNode.port.postMessage({
        type: 'audio',
        pcm: playbackPcm,
        responseId,
      }, [playbackPcm.buffer]);
      return;
    }
    if (event.type === 'response.audio_transcript.delta') {
      appendTranscript('assistant', event.delta || '');
      return;
    }
    if (event.type === 'conversation.item.input_audio_transcription.delta') {
      appendTranscript('user', event.delta || '');
      return;
    }
    if (event.type === 'response.function_call_arguments.done') {
      let args;
      try {
        args = JSON.parse(event.arguments || '{}');
      } catch (error) {
        log(`invalid tool arguments: ${error.message}`);
        return;
      }
      appendToolCall({
        id: event.call_id,
        name: event.name,
        arguments: args,
      });
      return;
    }
    if (
      event.type === 'conversation.item.created'
      && event.item
      && event.item.type === 'function_call_output'
    ) {
      appendToolResponse(event.item);
      return;
    }
    if (event.type === 'response.audio.done') {
      if (playbackNode) playbackNode.port.postMessage({ type: 'drain', responseId });
      return;
    }
    if (event.type === 'error') {
      const error = typeof event.error === 'string'
        ? event.error
        : JSON.stringify(event.error || event);
      detailElement.textContent = error;
      setStatus('Error', 'error');
      log(`error: ${error}`);
      return;
    }
    if (event.type) log(event.type);
  }

  async function stopSession() {
    const closingSocket = socket;
    socket = null;
    running = false;
    captureChunks = [];
    if (sendTimer !== null) clearInterval(sendTimer);
    sendTimer = null;
    if (recording) await stopRecording();
    if (closingSocket && closingSocket.readyState === WebSocket.OPEN) {
      closingSocket.onclose = null;
      closingSocket.send(JSON.stringify({ type: 'session.close' }));
      closingSocket.close(1000, 'client stop');
    }
    if (playbackNode) playbackNode.port.postMessage({ type: 'clear' });
    if (mediaStream) mediaStream.getTracks().forEach((track) => track.stop());
    if (captureContext) await captureContext.close().catch(() => {});
    if (playbackContext) await playbackContext.close().catch(() => {});
    mediaStream = null;
    captureContext = null;
    captureNode = null;
    playbackContext = null;
    playbackNode = null;
    recordingMicSource = null;
    recordingMerger = null;
    recordingNode = null;
    recordingSink = null;
    responseId = null;
    muted = false;
    meterFill.style.width = '0%';
    startButton.textContent = 'Start session';
    startButton.classList.remove('active');
    muteButton.textContent = 'Mute';
    muteButton.disabled = true;
    voiceSelect.disabled = false;
    samplingPicker.disabled = false;
    toolPicker.disabled = false;
    updateRecordingButton();
    setStatus('Offline');
    detailElement.textContent = 'No active connection';
  }

  async function startSession() {
    if (running) return;
    startButton.disabled = true;
    recordButton.disabled = true;
    voiceSelect.disabled = true;
    samplingPicker.disabled = true;
    toolPicker.disabled = true;
    setStatus('Starting');
    detailElement.textContent = 'Requesting microphone access';
    resetConversation();
    try {
      await openAudio();
      await openSocket();
      if (recordingArmed) startRecording();
      running = true;
      sendTimer = window.setInterval(flushCapture, sendIntervalMs);
      startButton.textContent = 'End session';
      startButton.classList.add('active');
      muteButton.disabled = false;
      setStatus('Connected', 'online');
      detailElement.textContent = 'Listening';
      log('session started');
    } catch (error) {
      log(`start failed: ${error.message || error}`);
      setStatus('Error', 'error');
      detailElement.textContent = error.message || String(error);
      await stopSession();
    } finally {
      startButton.disabled = false;
      recordButton.disabled = false;
    }
  }

  startButton.addEventListener('click', () => {
    if (running) stopSession();
    else startSession();
  });
  muteButton.addEventListener('click', () => {
    muted = !muted;
    captureChunks = [];
    muteButton.textContent = muted ? 'Unmute' : 'Mute';
    log(muted ? 'microphone muted' : 'microphone unmuted');
  });
  recordButton.addEventListener('click', async () => {
    if (recording) {
      recordingArmed = false;
      await stopRecording();
      return;
    }
    recordingArmed = !recordingArmed;
    if (running && recordingArmed) startRecording();
    else updateRecordingButton();
    log(recordingArmed ? 'session recording enabled' : 'session recording disabled');
  });
  populateVoices();
  populateSampling();
  populateTools();
  window.addEventListener('beforeunload', () => stopSession());
})();
