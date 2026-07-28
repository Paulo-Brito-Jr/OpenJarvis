const WAV_MIME_TYPE = 'audio/wav';
const MAX_WAV_BYTES = 16 * 1024 * 1024;

const SUPPORTED_RECORDING_MIME_TYPES = new Set([
  'audio/webm',
  'audio/ogg',
  'audio/mp4',
  'audio/mpeg',
]);

export interface AudioBufferLike {
  readonly length: number;
  readonly numberOfChannels: number;
  readonly sampleRate: number;
  getChannelData(channel: number): Float32Array;
}

const normalizedMimeType = (mimeType: string): string =>
  mimeType.split(';', 1)[0].trim().toLowerCase();

export const isSupportedRecordingMimeType = (mimeType: string): boolean =>
  SUPPORTED_RECORDING_MIME_TYPES.has(normalizedMimeType(mimeType));

const writeAscii = (view: DataView, offset: number, value: string): void => {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
};

/**
 * Encode decoded Web Audio samples as a mono RIFF/WAVE stream containing
 * little-endian signed 16-bit PCM. Compressed MediaRecorder bytes must be
 * decoded before calling this function; changing only their filename or MIME
 * type would not produce valid WAV audio.
 */
export function encodePcm16Wav(audio: AudioBufferLike): Blob {
  if (
    !Number.isInteger(audio.length)
    || audio.length <= 0
    || !Number.isInteger(audio.numberOfChannels)
    || audio.numberOfChannels <= 0
    || audio.numberOfChannels > 32
    || !Number.isInteger(audio.sampleRate)
    || !Number.isFinite(audio.sampleRate)
    || audio.sampleRate <= 0
    || audio.sampleRate > 384_000
  ) {
    throw new Error('Decoded audio has invalid PCM metadata');
  }

  const channels = Array.from(
    { length: audio.numberOfChannels },
    (_, channel) => audio.getChannelData(channel),
  );
  if (channels.some((channel) => channel.length !== audio.length)) {
    throw new Error('Decoded audio channel lengths do not match');
  }

  const channelCount = 1;
  const bitsPerSample = 16;
  const blockAlign = channelCount * (bitsPerSample / 8);
  const dataSize = audio.length * blockAlign;
  const byteLength = 44 + dataSize;
  if (!Number.isSafeInteger(byteLength) || byteLength > MAX_WAV_BYTES) {
    throw new Error('WAV audio exceeds the 16 MiB upload limit');
  }

  const buffer = new ArrayBuffer(byteLength);
  const view = new DataView(buffer);
  writeAscii(view, 0, 'RIFF');
  view.setUint32(4, byteLength - 8, true);
  writeAscii(view, 8, 'WAVE');
  writeAscii(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channelCount, true);
  view.setUint32(24, audio.sampleRate, true);
  view.setUint32(28, audio.sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  writeAscii(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  for (let frame = 0; frame < audio.length; frame += 1) {
    let mixedSample = 0;
    for (const channel of channels) {
      mixedSample += channel[frame];
    }
    mixedSample = Math.max(-1, Math.min(1, mixedSample / channels.length));
    const pcmSample = mixedSample < 0
      ? Math.round(mixedSample * 0x8000)
      : Math.round(mixedSample * 0x7fff);
    view.setInt16(44 + frame * blockAlign, pcmSample, true);
  }

  return new Blob([buffer], { type: WAV_MIME_TYPE });
}

const asciiAt = (bytes: Uint8Array, offset: number, value: string): boolean =>
  value.split('').every((character, index) => bytes[offset + index] === character.charCodeAt(0));

export function validatePcm16WavBytes(bytes: Uint8Array): void {
  if (bytes.byteLength < 44 || !asciiAt(bytes, 0, 'RIFF') || !asciiAt(bytes, 8, 'WAVE')) {
    throw new Error('Audio payload is not a RIFF/WAVE file');
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(4, true) + 8 !== bytes.byteLength) {
    throw new Error('WAV container length is invalid');
  }

  let offset = 12;
  let pcmFormat: { blockAlign: number; sampleRate: number } | null = null;
  let dataSize: number | null = null;
  while (offset + 8 <= bytes.byteLength) {
    const chunkSize = view.getUint32(offset + 4, true);
    const chunkDataOffset = offset + 8;
    const chunkEnd = chunkDataOffset + chunkSize;
    if (chunkEnd > bytes.byteLength) {
      throw new Error('WAV chunk exceeds the container length');
    }

    if (asciiAt(bytes, offset, 'fmt ')) {
      if (chunkSize < 16) {
        throw new Error('WAV format chunk is incomplete');
      }
      const audioFormat = view.getUint16(chunkDataOffset, true);
      const channelCount = view.getUint16(chunkDataOffset + 2, true);
      const sampleRate = view.getUint32(chunkDataOffset + 4, true);
      const byteRate = view.getUint32(chunkDataOffset + 8, true);
      const blockAlign = view.getUint16(chunkDataOffset + 12, true);
      const bitsPerSample = view.getUint16(chunkDataOffset + 14, true);
      const expectedBlockAlign = channelCount * 2;
      if (
        audioFormat !== 1
        || channelCount === 0
        || channelCount > 32
        || sampleRate === 0
        || sampleRate > 384_000
        || bitsPerSample !== 16
        || blockAlign !== expectedBlockAlign
        || byteRate !== sampleRate * expectedBlockAlign
      ) {
        throw new Error('WAV audio must contain 16-bit PCM samples');
      }
      if (pcmFormat) {
        throw new Error('WAV audio contains duplicate format chunks');
      }
      pcmFormat = { blockAlign, sampleRate };
    } else if (asciiAt(bytes, offset, 'data')) {
      if (chunkSize === 0) {
        throw new Error('WAV audio data is empty');
      }
      if (dataSize !== null) {
        throw new Error('WAV audio contains duplicate data chunks');
      }
      dataSize = chunkSize;
    }

    offset = chunkEnd + (chunkSize % 2);
  }

  if (offset !== bytes.byteLength) {
    throw new Error('WAV container has trailing or unpadded data');
  }
  if (!pcmFormat || dataSize === null) {
    throw new Error('WAV audio is missing required format or data chunks');
  }
  if (dataSize % pcmFormat.blockAlign !== 0) {
    throw new Error('WAV PCM data is not aligned to complete frames');
  }
  const frameCount = dataSize / pcmFormat.blockAlign;
  if (frameCount > pcmFormat.sampleRate * 15 * 60) {
    throw new Error('WAV audio exceeds the 15 minute duration limit');
  }
}

export async function readValidatedWav(
  audioBlob: Blob,
  filename: string,
): Promise<Uint8Array> {
  if (
    !filename
    || filename.includes('/')
    || filename.includes('\\')
    || !filename.toLowerCase().endsWith('.wav')
  ) {
    throw new Error('Transcription filename must be a WAV basename');
  }
  if (normalizedMimeType(audioBlob.type) !== WAV_MIME_TYPE) {
    throw new Error('Transcription audio must use the audio/wav MIME type');
  }
  if (audioBlob.size === 0 || audioBlob.size > MAX_WAV_BYTES) {
    throw new Error('WAV audio is empty or exceeds the 16 MiB upload limit');
  }

  const bytes = new Uint8Array(await audioBlob.arrayBuffer());
  validatePcm16WavBytes(bytes);
  return bytes;
}
