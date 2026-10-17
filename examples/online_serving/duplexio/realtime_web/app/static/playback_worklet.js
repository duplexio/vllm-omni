// Audio that arrives in a burst after a stall would otherwise stay queued as
// permanent extra latency, so silent frames that arrive while more than the
// playback buffer is queued are dropped.
const SILENCE_PEAK = 64;

class DuplexIOPlayback extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.queue = [];
    this.offset = 0;
    this.queuedFrames = 0;
    this.playedFrames = 0;
    this.responseId = null;
    this.drain = null;
    this.playing = false;
    this.underruns = 0;
    const { playbackBufferMs, maxPlaybackBufferMs, frameMs } = options.processorOptions;
    this.bufferFrames = Math.round(sampleRate * playbackBufferMs / 1000);
    this.maxBufferFrames = Math.round(sampleRate * maxPlaybackBufferMs / 1000);
    this.growFrames = Math.round(sampleRate * frameMs / 1000);
    this.port.onmessage = (event) => this.handle(event.data || {});
  }

  handle(message) {
    if (message.type === 'audio' && message.pcm) {
      if (!this.responseId) this.responseId = message.responseId || null;
      if (this.playing && this.queuedFrames > this.bufferFrames && isSilent(message.pcm)) {
        this.playedFrames += message.pcm.length;
        return;
      }
      this.queue.push(message.pcm);
      this.queuedFrames += message.pcm.length;
      return;
    }
    if (message.type === 'drain') {
      this.drain = { responseId: message.responseId || this.responseId };
      return;
    }
    if (message.type === 'clear') {
      this.queue = [];
      this.offset = 0;
      this.queuedFrames = 0;
      this.playedFrames = 0;
      this.responseId = null;
      this.drain = null;
      this.playing = false;
      this.underruns = 0;
    }
  }

  notifyDrained() {
    if (!this.drain || this.queue.length > 0) return;
    this.port.postMessage({
      type: 'drained',
      responseId: this.drain.responseId,
      playedMs: Math.round((this.playedFrames * 1000) / sampleRate),
    });
    this.playedFrames = 0;
    this.responseId = null;
    this.drain = null;
    this.playing = false;
    this.underruns = 0;
  }

  process(_inputs, outputs) {
    const output = outputs[0][0];
    output.fill(0);
    if (!this.playing) {
      if (this.queue.length === 0) {
        this.notifyDrained();
        return true;
      }
      if (this.queuedFrames < this.bufferFrames && !this.drain) {
        return true;
      }
      this.playing = true;
      this.port.postMessage({ type: 'started', responseId: this.responseId });
    }

    let target = 0;
    while (target < output.length && this.queue.length > 0) {
      const chunk = this.queue[0];
      const count = Math.min(output.length - target, chunk.length - this.offset);
      for (let index = 0; index < count; index += 1) {
        output[target + index] = chunk[this.offset + index] / 32768;
      }
      target += count;
      this.offset += count;
      this.queuedFrames -= count;
      this.playedFrames += count;
      if (this.offset === chunk.length) {
        this.queue.shift();
        this.offset = 0;
      }
    }
    if (this.queue.length === 0 && !this.drain) {
      this.playing = false;
      this.underruns += 1;
      // Kept for the rest of the session: a connection that stalled once
      // tends to stall again.
      this.bufferFrames = Math.min(this.maxBufferFrames, this.bufferFrames + this.growFrames);
      this.port.postMessage({
        type: 'buffering',
        responseId: this.responseId,
        underruns: this.underruns,
        bufferMs: Math.round((this.bufferFrames * 1000) / sampleRate),
      });
    }
    this.notifyDrained();
    return true;
  }
}

function isSilent(pcm) {
  for (let index = 0; index < pcm.length; index += 1) {
    if (Math.abs(pcm[index]) >= SILENCE_PEAK) return false;
  }
  return true;
}

registerProcessor('duplexio-playback', DuplexIOPlayback);
