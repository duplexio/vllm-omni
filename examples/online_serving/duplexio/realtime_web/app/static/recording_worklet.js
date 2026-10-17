class DuplexIORecorder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.recording = false;
    this.frames = 0;
    this.capacity = Math.max(128, Math.round(sampleRate));
    this.buffer = new Float32Array(this.capacity * 2);
    this.port.onmessage = (event) => this.handle(event.data || {});
  }

  handle(message) {
    if (message.type === 'start') {
      this.frames = 0;
      this.recording = true;
      return;
    }
    if (message.type === 'stop' && this.recording) {
      this.flush();
      this.recording = false;
      this.port.postMessage({ type: 'stopped', sampleRate });
    }
  }

  flush() {
    if (this.frames === 0) return;
    const pcm = this.buffer.slice(0, this.frames * 2);
    this.port.postMessage({ type: 'chunk', pcm: pcm.buffer }, [pcm.buffer]);
    this.frames = 0;
  }

  process(inputs, outputs) {
    const input = inputs[0] || [];
    const left = input[0];
    const right = input[1];
    const output = outputs[0] || [];
    for (const channel of output) channel.fill(0);
    if (!this.recording || (!left && !right)) return true;

    const frameCount = (left || right).length;
    for (let index = 0; index < frameCount; index += 1) {
      const target = this.frames * 2;
      this.buffer[target] = left ? left[index] : 0;
      this.buffer[target + 1] = right ? right[index] : 0;
      this.frames += 1;
      if (this.frames === this.capacity) this.flush();
    }
    return true;
  }
}

registerProcessor('duplexio-recorder', DuplexIORecorder);
