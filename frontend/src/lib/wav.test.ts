import { describe, expect, it } from 'vitest';
import {
  encodePcm16Wav,
  isSupportedRecordingMimeType,
  readValidatedWav,
  validatePcm16WavBytes,
} from './wav';

const decodedAudio = (
  channels: number[][],
  sampleRate = 16_000,
) => ({
  length: channels[0]?.length ?? 0,
  numberOfChannels: channels.length,
  sampleRate,
  getChannelData: (channel: number) => Float32Array.from(channels[channel]),
});

describe('encodePcm16Wav', () => {
  it('encodes decoded samples as mono little-endian PCM16 WAV', async () => {
    const wav = encodePcm16Wav(decodedAudio([
      [-1, 0, 1],
      [-1, 0.5, 1],
    ]));
    const bytes = new Uint8Array(await wav.arrayBuffer());
    const view = new DataView(bytes.buffer);

    expect(wav.type).toBe('audio/wav');
    expect(new TextDecoder().decode(bytes.slice(0, 4))).toBe('RIFF');
    expect(new TextDecoder().decode(bytes.slice(8, 12))).toBe('WAVE');
    expect(view.getUint16(20, true)).toBe(1);
    expect(view.getUint16(22, true)).toBe(1);
    expect(view.getUint32(24, true)).toBe(16_000);
    expect(view.getUint16(34, true)).toBe(16);
    expect(view.getInt16(44, true)).toBe(-32_768);
    expect(view.getInt16(46, true)).toBe(8_192);
    expect(view.getInt16(48, true)).toBe(32_767);
    expect(() => validatePcm16WavBytes(bytes)).not.toThrow();
  });

  it('rejects empty or inconsistent decoded audio', () => {
    expect(() => encodePcm16Wav(decodedAudio([]))).toThrow(/metadata/);
    expect(() => encodePcm16Wav({
      length: 2,
      numberOfChannels: 1,
      sampleRate: 16_000,
      getChannelData: () => Float32Array.from([0]),
    })).toThrow(/channel lengths/);
  });
});

describe('WAV upload validation', () => {
  it('accepts the generated PCM WAV with a WAV filename', async () => {
    const wav = encodePcm16Wav(decodedAudio([[0, 0.25, -0.25]]));
    await expect(readValidatedWav(wav, 'recording.wav')).resolves.toHaveLength(50);
  });

  it('rejects renamed compressed bytes and unsafe filenames', async () => {
    const renamedWebm = new Blob(
      [new TextEncoder().encode('not really wav')],
      { type: 'audio/wav' },
    );
    await expect(readValidatedWav(renamedWebm, 'recording.wav')).rejects.toThrow(/RIFF/);

    const wav = encodePcm16Wav(decodedAudio([[0]]));
    await expect(readValidatedWav(wav, '../recording.wav')).rejects.toThrow(/basename/);
    await expect(readValidatedWav(wav, 'recording.webm')).rejects.toThrow(/basename/);
  });

  it('recognizes only compressed formats the client can decode deliberately', () => {
    expect(isSupportedRecordingMimeType('audio/webm;codecs=opus')).toBe(true);
    expect(isSupportedRecordingMimeType('audio/mp4')).toBe(true);
    expect(isSupportedRecordingMimeType('video/webm')).toBe(false);
    expect(isSupportedRecordingMimeType('')).toBe(false);
  });
});
