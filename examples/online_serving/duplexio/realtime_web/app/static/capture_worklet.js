// Posts microphone audio in whole model frames, timed by the audio clock: a
// main-thread timer stalls under page load and runs once a second in a hidden
// tab, and every late frame comes back as a gap in the model's reply.
class DuplexIOCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.frame = new Int16Array(Math.round(sampleRate * options.processorOptions.frameMs / 1000));
    this.filled = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel || channel.length === 0) return true;

    for (let index = 0; index < channel.length; index += 1) {
      const sample = Math.max(-1, Math.min(1, channel[index]));
      this.frame[this.filled] = sample < 0 ? sample * 32768 : sample * 32767;
      this.filled += 1;
      if (this.filled === this.frame.length) {
        const frame = this.frame;
        this.frame = new Int16Array(frame.length);
        this.filled = 0;
        this.port.postMessage(frame.buffer, [frame.buffer]);
      }
    }
    return true;
  }
}

registerProcessor('duplexio-capture', DuplexIOCapture);
