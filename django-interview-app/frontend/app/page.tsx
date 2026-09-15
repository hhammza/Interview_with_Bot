"use client";

import { Mic, MicOff, Send, Square, Video } from "lucide-react";
import { useEffect, useRef, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api";
const BACKEND_BASE = API_BASE.replace(/\/api\/?$/, "");

type Message = { speaker: "assistant" | "user"; text: string };
type Metric = { dominant?: string; gaze?: string; lookAway?: boolean };

interface SpeechRecognitionResult {
  transcript: string;
}

interface SpeechRecognitionAlternativeList {
  [index: number]: SpeechRecognitionResult;
  isFinal: boolean;
}

interface SpeechRecognitionResultList {
  [index: number]: SpeechRecognitionAlternativeList;
  length: number;
}

interface SpeechRecognitionEvent {
  resultIndex: number;
  results: SpeechRecognitionResultList;
}

interface SpeechRecognition {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  onerror: ((event: { error: string }) => void) | null;
  onstart: (() => void) | null;
  onend: (() => void) | null;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  start: () => void;
  stop: () => void;
}

type SpeechSynthesisUtteranceCtor = new (text: string) => SpeechSynthesisUtterance;
type SpeechRecognitionCtor = new () => SpeechRecognition;
type BrowserWindow = Window & {
  SpeechRecognition?: SpeechRecognitionCtor;
  webkitSpeechRecognition?: SpeechRecognitionCtor;
  SpeechSynthesisUtterance?: SpeechSynthesisUtteranceCtor;
};

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const preflightVideoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const videoRecorderRef = useRef<MediaRecorder | null>(null);
  const audioRecorderRef = useRef<MediaRecorder | null>(null);
  const voiceRecorderRef = useRef<MediaRecorder | null>(null);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const answerTextRef = useRef("");
  const silenceTimerRef = useRef<number | null>(null);
  const videoChunksRef = useRef<Blob[]>([]);
  const audioChunksRef = useRef<Blob[]>([]);
  const sessionIdRef = useRef<string>("");
  const [jobRole, setJobRole] = useState("Software Engineer");
  const [sessionId, setSessionId] = useState("");
  const [stage, setStage] = useState<"preflight" | "countdown" | "interview">("preflight");
  const [cameraOk, setCameraOk] = useState(false);
  const [microphoneOk, setMicrophoneOk] = useState(false);
  const [countdown, setCountdown] = useState(5);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState("Ready");
  const [isFinished, setIsFinished] = useState(false);
  const [listening, setListening] = useState(false);
  const [draftAnswer, setDraftAnswer] = useState("");
  const [faceMetric, setFaceMetric] = useState<Metric>({});
  const [voiceMetric, setVoiceMetric] = useState<Metric>({});
  const [fused, setFused] = useState("neutral");
  const answeredCount = messages.filter(message => message.speaker === "user").length;
  const targetQuestions = 10;
  const progressPercent = Math.min((answeredCount / targetQuestions) * 100, 100);

  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);

  useEffect(() => {
    if (stage !== "interview" || !mediaStreamRef.current) return;
    if (videoRef.current) {
      videoRef.current.srcObject = mediaStreamRef.current;
    }
  }, [stage]);

  async function enableMedia() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      mediaStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      if (preflightVideoRef.current) {
        preflightVideoRef.current.srcObject = stream;
      }
      setStatus("Camera and microphone active");
    } catch (error) {
      setStatus("Camera or microphone permission was blocked");
      throw error;
    }
  }

  async function checkCameraAndMicrophone() {
    try {
      setStatus("Checking camera and microphone");
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      mediaStreamRef.current = stream;
      if (preflightVideoRef.current) {
        preflightVideoRef.current.srcObject = stream;
      }
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      const hasLiveVideo = stream.getVideoTracks().some(track => track.readyState === "live");
      setCameraOk(hasLiveVideo);

      const audioContext = new AudioContext();
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      const samples = new Uint8Array(analyser.frequencyBinCount);
      const startedAt = Date.now();
      let heardAudio = false;

      while (Date.now() - startedAt < 2500) {
        analyser.getByteFrequencyData(samples);
        const average = samples.reduce((sum, value) => sum + value, 0) / samples.length;
        if (average > 1) {
          heardAudio = true;
          break;
        }
        await new Promise(resolve => setTimeout(resolve, 150));
      }

      await audioContext.close();
      setMicrophoneOk(heardAudio);

      if (!hasLiveVideo || !heardAudio) {
        setStatus("Camera or microphone check failed. Speak briefly and try again.");
        return;
      }

      setStatus("Camera and microphone verified");
      setStage("countdown");
      setCountdown(5);
    } catch (error) {
      setCameraOk(false);
      setMicrophoneOk(false);
      setStatus("Camera or microphone permission was blocked");
    }
  }

  useEffect(() => {
    if (stage !== "countdown") return;
    if (countdown <= 0) {
      setStage("interview");
      void startInterview();
      return;
    }
    const timer = window.setTimeout(() => setCountdown(value => value - 1), 1000);
    return () => window.clearTimeout(timer);
  }, [stage, countdown]);

  function speakWithBrowser(text: string) {
    const SpeechUtterance = (window as BrowserWindow).SpeechSynthesisUtterance;
    if (!SpeechUtterance || !window.speechSynthesis) {
      setStatus("Speech is not supported in this browser");
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechUtterance(text);
    utterance.lang = "en-US";
    utterance.rate = 0.95;
    utterance.pitch = 1;
    utterance.onstart = () => setStatus("Interviewer speaking");
    utterance.onend = () => setStatus("Your turn");
    window.speechSynthesis.speak(utterance);
  }

  function playAssistantReply(text: string, dataUrl?: string) {
    if (!dataUrl) {
      speakWithBrowser(text);
      return;
    }
    const audio = new Audio(dataUrl);
    audio.onplay = () => setStatus("Interviewer speaking");
    audio.onended = () => setStatus("Your turn");
    audio.play().catch(() => speakWithBrowser(text));
  }

  function preferredMimeType(options: string[]) {
    return options.find(option => MediaRecorder.isTypeSupported(option)) || "";
  }

  async function startInterview() {
    try {
      setStatus("Starting interview");
      if (!mediaStreamRef.current) {
        await enableMedia();
      }
      const res = await fetch(`${API_BASE}/interviews/start/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_role: jobRole })
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Could not start interview");
      }
      setSessionId(data.sessionId);
      setMessages([{ speaker: "assistant", text: data.reply }]);
      playAssistantReply(data.reply, data.audioUrl);
      startRecording();
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not start interview";
      setStatus(message);
    }
  }

  function startRecording() {
    const stream = mediaStreamRef.current;
    if (!stream) return;
    videoChunksRef.current = [];
    audioChunksRef.current = [];

    const videoStream = new MediaStream(stream.getVideoTracks());
    const audioStream = new MediaStream(stream.getAudioTracks());
    const videoMimeType = preferredMimeType(["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"]);
    const audioMimeType = preferredMimeType(["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus"]);

    if (videoStream.getTracks().length) {
      const videoRecorder = new MediaRecorder(
        videoStream,
        videoMimeType ? { mimeType: videoMimeType } : undefined
      );
      videoRecorder.ondataavailable = event => {
        if (event.data.size > 0) videoChunksRef.current.push(event.data);
      };
      videoRecorderRef.current = videoRecorder;
      videoRecorder.start(1000);
    }

    if (audioStream.getTracks().length) {
      const audioRecorder = new MediaRecorder(
        audioStream,
        audioMimeType ? { mimeType: audioMimeType } : undefined
      );
      audioRecorder.ondataavailable = event => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data);
      };
      audioRecorderRef.current = audioRecorder;
      audioRecorder.start(1000);
    }
  }

  function stopRecorder(recorder: MediaRecorder | null) {
    return new Promise<void>(resolve => {
      if (!recorder || recorder.state === "inactive") {
        resolve();
        return;
      }
      recorder.onstop = () => resolve();
      recorder.stop();
    });
  }

  function downloadBlob(blob: Blob, filename: string) {
    const objectUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = objectUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(objectUrl);
  }

  async function submitText(text: string) {
    if (!sessionId || isFinished || !text.trim()) return;
    const clean = text.trim();
    setMessages(prev => [...prev, { speaker: "user", text: clean }]);
    setInput("");
    setStatus("Thinking");
    try {
      const res = await fetch(`${API_BASE}/interviews/${sessionId}/message/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: clean })
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Could not get interviewer response");
      }
      setMessages(prev => [...prev, { speaker: "assistant", text: data.reply }]);
      setFused(data.mood || fused);
      playAssistantReply(data.reply, data.audioUrl);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not get interviewer response";
      setStatus(message);
    }
  }

  function clearSilenceTimer() {
    if (silenceTimerRef.current) {
      window.clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
  }

  function scheduleAnswerSubmit() {
    clearSilenceTimer();
    silenceTimerRef.current = window.setTimeout(() => {
      finishListeningAnswer();
    }, 12000);
  }

  function finishListeningAnswer() {
    clearSilenceTimer();
    const recognition = recognitionRef.current;
    recognitionRef.current = null;
    if (recognition) {
      recognition.onend = null;
      recognition.stop();
    }
    setListening(false);
    const finalAnswer = answerTextRef.current.trim();
    answerTextRef.current = "";
    setDraftAnswer("");
    if (finalAnswer && !isFinished) {
      void submitText(finalAnswer);
    } else {
      setStatus("No answer captured");
    }
  }

  function listenForAnswer() {
    if (isFinished) return;
    if (listening) {
      finishListeningAnswer();
      return;
    }
    const BrowserSpeech = (window as BrowserWindow).SpeechRecognition || (window as BrowserWindow).webkitSpeechRecognition;
    if (!BrowserSpeech) {
      setStatus("Speech recognition is not supported in this browser");
      return;
    }
    const recognition = new BrowserSpeech();
    recognition.lang = "en-US";
    recognition.interimResults = true;
    recognition.continuous = true;
    recognitionRef.current = recognition;
    answerTextRef.current = "";
    setDraftAnswer("");
    recognition.onstart = () => {
      setListening(true);
      setStatus("Listening. Pause for 12 seconds or click Stop Answer when finished.");
      scheduleAnswerSubmit();
    };
    recognition.onend = () => {
      if (recognitionRef.current === recognition) {
        try {
          recognition.start();
        } catch {
          finishListeningAnswer();
        }
      }
    };
    recognition.onerror = event => {
      if (event.error === "no-speech") {
        scheduleAnswerSubmit();
        return;
      }
      setStatus(`Speech recognition error: ${event.error}`);
      finishListeningAnswer();
    };
    recognition.onresult = event => {
      let interim = "";
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const result = event.results[index];
        const text = result?.[0]?.transcript || "";
        if (result.isFinal) {
          answerTextRef.current = `${answerTextRef.current} ${text}`.trim();
        } else {
          interim = `${interim} ${text}`.trim();
        }
      }
      setDraftAnswer(`${answerTextRef.current} ${interim}`.trim());
      scheduleAnswerSubmit();
    };
    recognition.start();
  }

  useEffect(() => {
    if (!sessionId || isFinished) return;
    const timer = window.setInterval(async () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.videoWidth === 0) return;
      canvas.width = 480;
      canvas.height = Math.round((video.videoHeight / video.videoWidth) * 480);
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const image = canvas.toDataURL("image/jpeg", 0.75);
      const res = await fetch(`${API_BASE}/interviews/${sessionIdRef.current}/face-frame/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image })
      });
      if (res.ok) {
        const data = await res.json();
        setFaceMetric({ dominant: data.dominant, gaze: data.gaze, lookAway: data.lookAway });
        setFused(data.fused?.dominant || "neutral");
      }
    }, 1400);
    return () => window.clearInterval(timer);
  }, [sessionId, isFinished]);

  useEffect(() => {
    if (!sessionId || !mediaStreamRef.current || isFinished) return;
    const stream = mediaStreamRef.current;
    const timer = window.setInterval(() => {
      if (voiceRecorderRef.current?.state === "recording") return;
      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      const audioChunks: Blob[] = [];
      recorder.ondataavailable = event => {
        if (event.data.size > 0) audioChunks.push(event.data);
      };
      recorder.onstop = async () => {
        const blob = new Blob(audioChunks, { type: "audio/webm" });
        const form = new FormData();
        form.append("audio", blob, "voice.webm");
        const res = await fetch(`${API_BASE}/interviews/${sessionIdRef.current}/voice-sample/`, {
          method: "POST",
          body: form
        });
        if (res.ok) {
          const data = await res.json();
          setVoiceMetric({ dominant: data.dominant });
          setFused(data.fused?.dominant || "neutral");
        }
      };
      voiceRecorderRef.current = recorder;
      recorder.start();
      window.setTimeout(() => recorder.state === "recording" && recorder.stop(), 3000);
    }, 5000);
    return () => window.clearInterval(timer);
  }, [sessionId, isFinished]);

  async function finishInterview() {
    if (!sessionId || isFinished) return;
    setIsFinished(true);
    setListening(false);
    clearSilenceTimer();
    if (recognitionRef.current) {
      recognitionRef.current.onend = null;
      recognitionRef.current.stop();
      recognitionRef.current = null;
    }
    window.speechSynthesis?.cancel();
    setStatus("Stopping interview and preparing download bundle");

    await Promise.all([
      stopRecorder(videoRecorderRef.current),
      stopRecorder(audioRecorderRef.current),
      stopRecorder(voiceRecorderRef.current)
    ]);

    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const videoBlob = videoChunksRef.current.length
      ? new Blob(videoChunksRef.current, { type: videoChunksRef.current[0]?.type || "video/webm" })
      : null;
    const audioBlob = audioChunksRef.current.length
      ? new Blob(audioChunksRef.current, { type: audioChunksRef.current[0]?.type || "audio/webm" })
      : null;

    if (videoBlob || audioBlob) {
      const form = new FormData();
      if (videoBlob) {
        form.append("video", videoBlob, `interview-camera-${timestamp}.webm`);
      }
      if (audioBlob) {
        form.append("audio", audioBlob, `interview-microphone-${timestamp}.webm`);
      }
      await fetch(`${API_BASE}/interviews/${sessionId}/recording/`, { method: "POST", body: form });
    }

    const res = await fetch(`${API_BASE}/interviews/${sessionId}/finish/`, { method: "POST" });
    const data = await res.json();
    if (!res.ok) {
      setStatus(data.detail || "Could not finish interview");
      return;
    }

    const bundleUrl = data.bundleDownloadUrl || data.transcriptDownloadUrl || data.transcriptUrl;
    const downloadUrl = bundleUrl.startsWith("http")
      ? bundleUrl
      : `${BACKEND_BASE}${bundleUrl}`;
    const bundleRes = await fetch(downloadUrl);
    if (!bundleRes.ok) {
      setStatus("Interview was saved, but the bundle download failed");
      return;
    }
    const bundleBlob = await bundleRes.blob();
    downloadBlob(bundleBlob, data.bundleFilename || `interview-${sessionId}.zip`);

    mediaStreamRef.current?.getTracks().forEach(track => track.stop());
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setStatus("Finished. Interview bundle downloaded.");
  }

  if (stage === "preflight" || stage === "countdown") {
    return (
      <main className="preflightShell">
        <section className="preflightPanel">
          <div className="preflightHeader">
            <h1>Interview Setup</h1>
            <p>Camera and microphone access is required before the interview begins.</p>
          </div>
          <div className="preflightGrid">
            <div className="preflightPreview">
              <video ref={preflightVideoRef} autoPlay playsInline muted />
            </div>
            <div className="preflightControls">
              <label className="fieldLabel" htmlFor="job-role">Job role</label>
              <input id="job-role" value={jobRole} onChange={event => setJobRole(event.target.value)} />
              <div className="checkList">
                <div className={cameraOk ? "checkItem ok" : "checkItem"}>
                  <span>Camera</span>
                  <strong>{cameraOk ? "Working" : "Not checked"}</strong>
                </div>
                <div className={microphoneOk ? "checkItem ok" : "checkItem"}>
                  <span>Microphone</span>
                  <strong>{microphoneOk ? "Working" : "Not checked"}</strong>
                </div>
              </div>
              {stage === "countdown" ? (
                <div className="countdownBox">
                  <span>Interview starts in</span>
                  <strong>{countdown}</strong>
                </div>
              ) : (
                <button className="primary" onClick={checkCameraAndMicrophone}>
                  <Video size={18} /> Check Camera and Mic
                </button>
              )}
              <div className="status">{status}</div>
            </div>
          </div>
        </section>
      </main>
    );
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand">Interview Emotion Assistant</div>
        <div className="progressWrap" aria-label="Interview progress">
          <span>{answeredCount}/{targetQuestions} answered</span>
          <div className="progressTrack">
            <div className="progressFill" style={{ width: `${progressPercent}%` }} />
          </div>
        </div>
        <div className="status">{status}</div>
      </header>
      <section className="workspace">
        <div className="videoPane">
          <div className="videoWrap">
            <video ref={videoRef} autoPlay playsInline muted />
            {faceMetric.lookAway && <div className="warning">LOOK AT THE CAMERA</div>}
          </div>
          <div className="setup">
            <div className="rolePill">Role: {jobRole}</div>
            <button onClick={finishInterview} disabled={!sessionId || isFinished} title="Finish and save interview files">
              <Square size={18} /> Finish
            </button>
          </div>
        </div>
        <div className="sidePane">
          <div className="sideHeader">
            <h1>{jobRole}</h1>
            <p>Live interview session</p>
          </div>
          <div className="messages">
            {messages.map((message, index) => (
              <div key={`${message.speaker}-${index}`} className={`bubble ${message.speaker}`}>
                {message.text}
              </div>
            ))}
          </div>
          <div className="composer">
            <button onClick={listenForAnswer} disabled={!sessionId || isFinished} className="secondary" title="Answer by voice">
              {listening ? <MicOff size={18} /> : <Mic size={18} />} {listening ? "Stop Answer" : "Speak"}
            </button>
            <input value={input} disabled={isFinished} onChange={event => setInput(event.target.value)} onKeyDown={event => event.key === "Enter" && submitText(input)} aria-label="Typed answer" />
            <button onClick={() => submitText(input)} disabled={!sessionId || isFinished || !input.trim()} title="Send answer">
              <Send size={18} /> Send
            </button>
          </div>
          {draftAnswer && <div className="draftAnswer">{draftAnswer}</div>}
        </div>
      </section>
      <canvas ref={canvasRef} hidden />
    </main>
  );
}
