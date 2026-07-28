import { useState, useCallback, useRef, useEffect } from 'react';
import { transcribeAudio, fetchSpeechHealth } from '../lib/api';
import {
  encodePcm16Wav,
  isSupportedRecordingMimeType,
} from '../lib/wav';

export type SpeechState = 'idle' | 'recording' | 'transcribing';

export function useSpeech() {
  const [state, setState] = useState<SpeechState>('idle');
  const [error, setError] = useState<string | null>(null);
  const [available, setAvailable] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  // Check if speech backend is available on mount
  useEffect(() => {
    fetchSpeechHealth()
      .then((health) => setAvailable(health.available))
      .catch(() => setAvailable(false));
  }, []);

  const startRecording = useCallback(async (): Promise<void> => {
    setError(null);

    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      setError('Microphone not supported in this browser');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const preferredMimeType = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/ogg;codecs=opus',
        'audio/mp4',
      ].find((mimeType) => MediaRecorder.isTypeSupported(mimeType));
      const recorder = preferredMimeType
        ? new MediaRecorder(stream, { mimeType: preferredMimeType })
        : new MediaRecorder(stream);
      if (!isSupportedRecordingMimeType(recorder.mimeType)) {
        stream.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        throw new Error(`Unsupported recording format: ${recorder.mimeType || 'unknown'}`);
      }
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.start();
      mediaRecorderRef.current = recorder;
      setState('recording');
    } catch (err) {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      const message = err instanceof Error ? err.message : 'Microphone access denied';
      setError(message);
      setState('idle');
    }
  }, []);

  const stopRecording = useCallback(async (): Promise<string> => {
    return new Promise((resolve, reject) => {
      const recorder = mediaRecorderRef.current;
      if (!recorder || recorder.state !== 'recording') {
        reject(new Error('Not recording'));
        return;
      }

      recorder.onstop = async () => {
        mediaRecorderRef.current = null;
        setState('transcribing');

        // Stop all audio tracks
        streamRef.current?.getTracks().forEach((track) => track.stop());
        streamRef.current = null;

        const recordedBlob = new Blob(chunksRef.current, { type: recorder.mimeType });
        chunksRef.current = [];

        try {
          if (recordedBlob.size === 0) {
            throw new Error('Recorded audio is empty');
          }
          const AudioContextConstructor = window.AudioContext
            || (window as unknown as { webkitAudioContext?: typeof AudioContext })
              .webkitAudioContext;
          if (!AudioContextConstructor) {
            throw new Error('Audio decoding is not supported in this browser');
          }

          const audioContext = new AudioContextConstructor();
          let decodedAudio: AudioBuffer;
          try {
            decodedAudio = await audioContext.decodeAudioData(
              await recordedBlob.arrayBuffer(),
            );
          } catch {
            throw new Error('Recorded audio could not be decoded');
          } finally {
            await audioContext.close().catch(() => undefined);
          }

          const wavBlob = encodePcm16Wav(decodedAudio);
          const result = await transcribeAudio(wavBlob, 'recording.wav');
          setState('idle');
          resolve(result.text);
        } catch (err) {
          setState('idle');
          const msg = err instanceof Error ? err.message : 'Transcription failed';
          setError(msg);
          reject(err);
        }
      };

      recorder.stop();
    });
  }, []);

  return {
    state,
    error,
    available,
    startRecording,
    stopRecording,
    isRecording: state === 'recording',
    isTranscribing: state === 'transcribing',
  };
}
