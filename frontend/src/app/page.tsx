"use client";

import React, { useState, useEffect, useRef } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import {
  Play,
  Pause,
  Loader2,
  ChevronRight,
  Download,
  Sliders,
  Activity,
  Layers,
  ArrowLeft,
  RefreshCw,
  Clock,
  TrendingUp,
  Cpu
} from "lucide-react";

// API Base URL - default to local dev server
const API_BASE = "http://localhost:8000";

// Interface Definitions
interface DatasetInfo {
  modality: string;
  filename: string;
  size_bytes: number;
}

interface SceneMetadata {
  scene_id: string;
  modality: string;
  min_raw_val: number;
  max_raw_val: number;
  shape: number[];
  frames: number;
  [key: string]: any;
}

export default function Home() {
  const [view, setView] = useState<"landing" | "workspace">("landing");
  
  // Workspace States
  const [selectedDataset, setSelectedDataset] = useState<string>("sevir");
  const [selectedModality, setSelectedModality] = useState<string>("vis");
  const [selectedScene, setSelectedScene] = useState<string>("");
  
  // Timeline States
  const [frameBefore, setFrameBefore] = useState<number>(23);
  const [frameAfter, setFrameAfter] = useState<number>(25);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [currentFrameIndex, setCurrentFrameIndex] = useState<number>(23);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(250); // ms per frame
  
  // Prediction / Sequence states
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const [sequenceResults, setSequenceResults] = useState<any[]>([]);
  const [numSequenceFrames, setNumSequenceFrames] = useState<number>(3);
  const [isInterpolating, setIsInterpolating] = useState<boolean>(false);
  const [compareMode, setCompareMode] = useState<"side" | "slider">("slider");
  const [sliderPosition, setSliderPosition] = useState<number>(50); // percentage split
  const sliderRef = useRef<HTMLDivElement>(null);
  
  // API Queries using React Query
  const { data: datasetsData, isLoading: loadingDatasets } = useQuery({
    queryKey: ["datasets"],
    queryFn: async () => {
      const res = await fetch(`${API_BASE}/api/v1/datasets`);
      if (!res.ok) throw new Error("Failed to fetch datasets");
      return res.json();
    }
  });

  const { data: scenesData, isLoading: loadingScenes } = useQuery({
    queryKey: ["scenes", selectedDataset, selectedModality],
    queryFn: async () => {
      const res = await fetch(
        `${API_BASE}/api/v1/datasets/${selectedDataset}/scenes?modality=${selectedModality}`
      );
      if (!res.ok) throw new Error("Failed to fetch scenes");
      return res.json();
    },
    enabled: !!selectedDataset && !!selectedModality
  });

  const { data: sceneDetail, isLoading: loadingSceneDetail } = useQuery({
    queryKey: ["scene", selectedScene],
    queryFn: async () => {
      const res = await fetch(`${API_BASE}/api/v1/scenes/${selectedScene}`);
      if (!res.ok) throw new Error("Failed to fetch scene details");
      return res.json();
    },
    enabled: !!selectedScene
  });

  // Automatically select first scene when list loads
  useEffect(() => {
    if (scenesData?.scenes && scenesData.scenes.length > 0) {
      setSelectedScene(scenesData.scenes[0]);
    }
  }, [scenesData]);

  // Set default timeline limits when scene metadata is available
  useEffect(() => {
    if (sceneDetail?.properties?.frames) {
      const total = sceneDetail.properties.frames;
      setFrameBefore(Math.floor(total / 2) - 1);
      setFrameAfter(Math.floor(total / 2) + 1);
      setCurrentFrameIndex(Math.floor(total / 2) - 1);
    }
  }, [sceneDetail]);

  // Handle Playback Interval Animation Loop
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isPlaying) {
      interval = setInterval(() => {
        setCurrentFrameIndex((prev) => {
          if (prev >= frameAfter) return frameBefore;
          return prev + 1;
        });
      }, playbackSpeed);
    }
    return () => clearInterval(interval);
  }, [isPlaying, frameBefore, frameAfter, playbackSpeed]);

  // Trigger Interpolation Mutation
  const interpolateMutation = useMutation({
    mutationFn: async () => {
      setIsInterpolating(true);
      setPredictionResult(null);
      const res = await fetch(`${API_BASE}/api/v1/interpolate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scene_id: selectedScene,
          modality: selectedModality,
          frame_before: frameBefore,
          frame_after: frameAfter
        })
      });
      if (!res.ok) throw new Error("Interpolation failed");
      return res.json();
    },
    onSuccess: (data) => {
      setPredictionResult(data);
      setIsInterpolating(false);
    },
    onError: () => {
      setIsInterpolating(false);
    }
  });

  // Trigger Sequence Interpolation Mutation
  const interpolateSeqMutation = useMutation({
    mutationFn: async () => {
      setIsInterpolating(true);
      setSequenceResults([]);
      const res = await fetch(`${API_BASE}/api/v1/interpolate-sequence`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scene_id: selectedScene,
          modality: selectedModality,
          frame_before: frameBefore,
          frame_after: frameAfter,
          num_frames: numSequenceFrames
        })
      });
      if (!res.ok) throw new Error("Sequence interpolation failed");
      return res.json();
    },
    onSuccess: (data) => {
      setSequenceResults(data);
      setIsInterpolating(false);
    },
    onError: () => {
      setIsInterpolating(false);
    }
  });

  // Handle Slider Mouse Drag
  const handleSliderMove = (clientX: number) => {
    if (!sliderRef.current) return;
    const rect = sliderRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPosition(percentage);
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    if (e.touches.length > 0) {
      handleSliderMove(e.touches[0].clientX);
    }
  };

  // Helper to construct image urls from backend FastAPI server
  const getFrameUrl = (idx: number) => {
    return `${API_BASE}/api/v1/scenes/${selectedScene}/frames/${idx}/png`;
  };

  const getPredictionUrl = () => {
    if (!predictionResult?.saved_to) return "";
    // Pull the basename of the saved file to serve from static mount
    const filename = predictionResult.saved_to.split(/[\\/]/).pop();
    return `${API_BASE}/static/${filename}`;
  };

  // Render Page Content
  return (
    <div className="flex-1 flex flex-col font-sans">
      <AnimatePresence mode="wait">
        {view === "landing" ? (
          // ================= LANDING PAGE =================
          <motion.div
            key="landing"
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -15 }}
            className="flex-1 flex flex-col items-center justify-center p-6 text-center max-w-5xl mx-auto"
          >
            {/* Background Glow */}
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(99,102,241,0.08),transparent_50%)] pointer-events-none" />

            <div className="space-y-6 relative">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-sm font-semibold tracking-wide">
                <Cpu className="w-4 h-4 animate-pulse" /> Geostationary Temporal Intelligence
              </div>

              <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight bg-gradient-to-r from-zinc-100 via-zinc-200 to-indigo-400 bg-clip-text text-transparent">
                FrameSat AI
              </h1>
              
              <p className="text-lg md:text-xl text-zinc-400 max-w-2xl mx-auto leading-relaxed">
                Reconstruct high-fidelity intermediate environmental observations inside geostationary gaps using motion-aware deep interpolation models.
              </p>

              {/* Grid cards detailing features */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-10 text-left">
                <div className="p-6 rounded-2xl bg-zinc-900/40 border border-zinc-800/80 backdrop-blur-md space-y-3">
                  <div className="w-10 h-10 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400">
                    <Layers className="w-5 h-5" />
                  </div>
                  <h3 className="font-bold text-zinc-200">Multimodal Datasets</h3>
                  <p className="text-sm text-zinc-400">Support for SEVIR VIS, VIL, IR, and raw GOES-19 weather channels natively mapped.</p>
                </div>

                <div className="p-6 rounded-2xl bg-zinc-900/40 border border-zinc-800/80 backdrop-blur-md space-y-3">
                  <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400">
                    <Clock className="w-5 h-5" />
                  </div>
                  <h3 className="font-bold text-zinc-200">Recursive Spacing</h3>
                  <p className="text-sm text-zinc-400">Generate configurable sequences (1, 2, 3, 7, 15 frames) with uniform temporal spacing.</p>
                </div>

                <div className="p-6 rounded-2xl bg-zinc-900/40 border border-zinc-800/80 backdrop-blur-md space-y-3">
                  <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
                    <TrendingUp className="w-5 h-5" />
                  </div>
                  <h3 className="font-bold text-zinc-200">Validation Analytics</h3>
                  <p className="text-sm text-zinc-400">Extract ground truth comparisons with full PSNR, SSIM, and difference heatmaps.</p>
                </div>
              </div>

              {/* Call to action */}
              <div className="pt-8">
                <button
                  onClick={() => setView("workspace")}
                  className="group relative inline-flex items-center gap-2 px-8 py-4 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold transition-all duration-300 shadow-[0_0_20px_rgba(99,102,241,0.3)] hover:shadow-[0_0_30px_rgba(99,102,241,0.5)]"
                >
                  Launch Analysis Workspace
                  <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                </button>
              </div>
            </div>
          </motion.div>
        ) : (
          // ================= WORKSPACE VIEW =================
          <motion.div
            key="workspace"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex-1 flex flex-col h-screen"
          >
            {/* Workspace Header Nav */}
            <header className="flex items-center justify-between px-6 py-4 border-b border-zinc-800/80 bg-zinc-950/80 backdrop-blur-md z-10">
              <div className="flex items-center gap-4">
                <button
                  onClick={() => setView("landing")}
                  className="p-2 rounded-lg hover:bg-zinc-900 text-zinc-400 hover:text-zinc-100 transition-colors"
                >
                  <ArrowLeft className="w-5 h-5" />
                </button>
                <div className="flex items-center gap-2">
                  <span className="font-bold text-zinc-200">FrameSat AI</span>
                  <span className="text-xs px-2 py-0.5 rounded bg-zinc-800 text-zinc-400">v1.2.0</span>
                </div>
              </div>

              <div className="flex items-center gap-6 text-sm">
                <div className="flex items-center gap-2 text-zinc-400">
                  <Clock className="w-4 h-4 text-emerald-400" />
                  <span>API Status: </span>
                  <span className="font-semibold text-emerald-400">Online</span>
                </div>
              </div>
            </header>

            {/* Main Panel Grid Layout */}
            <div className="flex-1 grid grid-cols-1 xl:grid-cols-4 overflow-hidden">
              
              {/* LEFT PANEL: Control center */}
              <aside className="border-r border-zinc-800/80 bg-zinc-950 p-6 flex flex-col gap-6 overflow-y-auto">
                <div className="space-y-4">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-500">Dataset Configuration</h3>
                  
                  {/* Dataset modality choice */}
                  <div className="space-y-2">
                    <label className="text-xs text-zinc-400">Target Source</label>
                    <select
                      value={selectedDataset}
                      onChange={(e) => {
                        setSelectedDataset(e.target.value);
                        setSelectedModality(e.target.value === "goes19" ? "goes_c13" : "vis");
                      }}
                      className="w-full px-3 py-2 rounded-lg bg-zinc-900 border border-zinc-800 focus:outline-none focus:border-indigo-500 text-sm text-zinc-200"
                    >
                      <option value="sevir">SEVIR storm events</option>
                      <option value="goes19">GOES-19 GOES19Provider</option>
                    </select>
                  </div>

                  <div className="space-y-2">
                    <label className="text-xs text-zinc-400">Sensor Modality</label>
                    <select
                      value={selectedModality}
                      onChange={(e) => setSelectedModality(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg bg-zinc-900 border border-zinc-800 focus:outline-none focus:border-indigo-500 text-sm text-zinc-200"
                    >
                      {selectedDataset === "sevir" ? (
                        <>
                          <option value="vis">VIS (Visible Band)</option>
                          <option value="vil">VIL (Radar Composite)</option>
                        </>
                      ) : (
                        <option value="goes_c13">ABI Channel 13 (10.3 µm)</option>
                      )}
                    </select>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-500">Active Scene</h3>
                    {loadingScenes && <Loader2 className="w-3.5 h-3.5 animate-spin text-zinc-400" />}
                  </div>

                  <select
                    value={selectedScene}
                    onChange={(e) => setSelectedScene(e.target.value)}
                    disabled={loadingScenes || !scenesData?.scenes}
                    className="w-full px-3 py-2 rounded-lg bg-zinc-900 border border-zinc-800 focus:outline-none focus:border-indigo-500 text-sm text-zinc-200 disabled:opacity-50"
                  >
                    {scenesData?.scenes?.map((scene: string) => (
                      <option key={scene} value={scene}>{scene}</option>
                    ))}
                  </select>
                </div>

                {/* Timeline endpoints */}
                <div className="space-y-4 pt-4 border-t border-zinc-800/80">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-500">Frame Range</h3>
                  
                  {loadingSceneDetail ? (
                    <div className="flex items-center justify-center p-4">
                      <Loader2 className="w-6 h-6 animate-spin text-indigo-500" />
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <div className="flex justify-between text-xs">
                          <span className="text-zinc-400">Start Frame (t0)</span>
                          <span className="font-semibold text-zinc-200">{frameBefore}</span>
                        </div>
                        <input
                          type="range"
                          min={0}
                          max={frameAfter - 2}
                          value={frameBefore}
                          onChange={(e) => {
                            const val = parseInt(e.target.value);
                            setFrameBefore(val);
                            setCurrentFrameIndex(val);
                          }}
                          className="w-full accent-indigo-500"
                        />
                      </div>

                      <div className="space-y-2">
                        <div className="flex justify-between text-xs">
                          <span className="text-zinc-400">End Frame (t2)</span>
                          <span className="font-semibold text-zinc-200">{frameAfter}</span>
                        </div>
                        <input
                          type="range"
                          min={frameBefore + 2}
                          max={(sceneDetail?.properties?.frames || 49) - 1}
                          value={frameAfter}
                          onChange={(e) => {
                            const val = parseInt(e.target.value);
                            setFrameAfter(val);
                          }}
                          className="w-full accent-indigo-500"
                        />
                      </div>

                      <div className="p-3 rounded-lg bg-zinc-900 border border-zinc-800 text-xs space-y-1">
                        <div className="flex justify-between">
                          <span className="text-zinc-400">Temporal Gap:</span>
                          <span className="text-zinc-200 font-semibold">{(frameAfter - frameBefore) * 5} mins</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-zinc-400">Midpoint Frame (t1):</span>
                          <span className="text-zinc-200 font-semibold">{Math.floor((frameBefore + frameAfter) / 2)}</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </aside>

              {/* CENTER PANEL: Main Frame Viewer */}
              <main className="xl:col-span-2 bg-zinc-900/30 flex flex-col overflow-hidden">
                <div className="flex-1 flex flex-col items-center justify-center p-6 overflow-hidden relative">
                  
                  {/* Selection Info */}
                  <div className="absolute top-6 left-6 z-10">
                    <span className="text-xs px-2.5 py-1 rounded-full bg-zinc-950/80 border border-zinc-800 text-zinc-400 backdrop-blur-sm">
                      {selectedScene} | Frame {currentFrameIndex}
                    </span>
                  </div>

                  {/* Frame Display / Interactive Comparison Slider */}
                  <div className="w-full max-w-xl aspect-square bg-zinc-950 rounded-2xl border border-zinc-800 overflow-hidden relative shadow-2xl">
                    {selectedScene ? (
                      compareMode === "slider" && predictionResult ? (
                        /* Image comparison slider */
                        <div
                          ref={sliderRef}
                          onMouseMove={(e) => handleSliderMove(e.clientX)}
                          onTouchMove={handleTouchMove}
                          className="w-full h-full relative cursor-ew-resize select-none"
                        >
                          {/* Ground Truth underneath */}
                          <img
                            src={getFrameUrl(Math.floor((frameBefore + frameAfter) / 2))}
                            alt="Ground Truth"
                            className="w-full h-full object-cover pointer-events-none"
                          />

                          {/* Prediction layer on top with dynamic clip-path */}
                          <div
                            className="absolute inset-0 pointer-events-none"
                            style={{ clipPath: `polygon(0 0, ${sliderPosition}% 0, ${sliderPosition}% 100%, 0 100%)` }}
                          >
                            <img
                              src={getPredictionUrl()}
                              alt="Prediction"
                              className="w-full h-full object-cover"
                            />
                          </div>

                          {/* Labels placed outside the clipped container */}
                          <div className="absolute top-4 left-4 z-10 px-2 py-0.5 rounded bg-indigo-500/80 text-xs font-semibold text-white pointer-events-none">
                            Prediction
                          </div>
                          <div className="absolute top-4 right-4 z-10 px-2 py-0.5 rounded bg-zinc-900/80 text-xs font-semibold text-zinc-300 pointer-events-none">
                            Ground Truth
                          </div>

                          {/* Sliding Bar Handle */}
                          <div
                            className="absolute top-0 bottom-0 w-0.5 bg-indigo-500 z-20 pointer-events-none"
                            style={{ left: `${sliderPosition}%` }}
                          >
                            <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-indigo-600 border-2 border-white flex items-center justify-center shadow-lg">
                              <Sliders className="w-4 h-4 text-white rotate-90" />
                            </div>
                          </div>
                        </div>
                      ) : (
                        /* Standard timeline player mode */
                        <img
                          src={getFrameUrl(currentFrameIndex)}
                          alt={`Frame ${currentFrameIndex}`}
                          className="w-full h-full object-cover"
                        />
                      )
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-zinc-500 text-sm">
                        Select a scene to start loading frames.
                      </div>
                    )}
                  </div>

                  {/* Playback Controls */}
                  <div className="w-full max-w-md mt-6 p-4 rounded-xl bg-zinc-950/80 border border-zinc-800/80 backdrop-blur-md flex items-center justify-between gap-4">
                    <button
                      onClick={() => setIsPlaying(!isPlaying)}
                      className="p-3 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white transition-colors"
                    >
                      {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                    </button>

                    <div className="flex-1 flex flex-col gap-1">
                      <div className="flex justify-between text-[10px] text-zinc-500">
                        <span>Speed</span>
                        <span>{playbackSpeed} ms</span>
                      </div>
                      <input
                        type="range"
                        min={100}
                        max={1000}
                        step={50}
                        value={playbackSpeed}
                        onChange={(e) => setPlaybackSpeed(parseInt(e.target.value))}
                        className="w-full accent-indigo-500"
                      />
                    </div>
                  </div>
                </div>
              </main>

              {/* RIGHT PANEL: Results panel */}
              <aside className="border-l border-zinc-800/80 bg-zinc-950 p-6 flex flex-col gap-6 overflow-y-auto">
                <div className="space-y-4">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-500">AI Interpolation Engine</h3>
                  
                  {/* Single frame prediction */}
                  <button
                    onClick={() => interpolateMutation.mutate()}
                    disabled={isInterpolating || !selectedScene}
                    className="w-full py-3 px-4 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-sm transition-all duration-200 flex items-center justify-center gap-2 disabled:opacity-50 shadow-md"
                  >
                    {isInterpolating ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Generating Frame...
                      </>
                    ) : (
                      <>
                        <RefreshCw className="w-4 h-4" />
                        Generate Midpoint Frame
                      </>
                    )}
                  </button>

                  {/* Multi sequence generation */}
                  <div className="pt-4 border-t border-zinc-800/80 space-y-3">
                    <div className="flex justify-between items-center text-xs">
                      <span className="text-zinc-400">Sequence frames</span>
                      <select
                        value={numSequenceFrames}
                        onChange={(e) => setNumSequenceFrames(parseInt(e.target.value))}
                        className="px-2 py-1 rounded bg-zinc-900 border border-zinc-800 text-zinc-300 text-xs"
                      >
                        <option value={1}>1 Frame (0.50)</option>
                        <option value={2}>2 Frames (0.25, 0.75)</option>
                        <option value={3}>3 Frames (0.25, 0.50, 0.75)</option>
                        <option value={7}>7 Frames (0.125 increments)</option>
                        <option value={15}>15 Frames (0.0625 increments)</option>
                      </select>
                    </div>

                    <button
                      onClick={() => interpolateSeqMutation.mutate()}
                      disabled={isInterpolating || !selectedScene}
                      className="w-full py-2.5 px-4 rounded-lg bg-zinc-900 border border-zinc-800 hover:bg-zinc-800 text-zinc-200 text-xs font-medium transition-all flex items-center justify-center gap-2 disabled:opacity-50"
                    >
                      {isInterpolating ? (
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      ) : (
                        "Generate Sequence"
                      )}
                    </button>
                  </div>
                </div>

                {/* Metrics Dashboard */}
                <div className="space-y-4 pt-4 border-t border-zinc-800/80">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-500">Evaluation Metrics</h3>
                  
                  {predictionResult ? (
                    <div className="grid grid-cols-2 gap-3">
                      <div className="p-3 rounded-lg bg-zinc-900/60 border border-zinc-800/80">
                        <div className="text-[10px] text-zinc-500 uppercase">PSNR</div>
                        <div className="text-lg font-bold text-zinc-200">{predictionResult.metrics?.psnr?.toFixed(2)} dB</div>
                      </div>

                      <div className="p-3 rounded-lg bg-zinc-900/60 border border-zinc-800/80">
                        <div className="text-[10px] text-zinc-500 uppercase">SSIM</div>
                        <div className="text-lg font-bold text-zinc-200">{predictionResult.metrics?.ssim?.toFixed(4)}</div>
                      </div>

                      <div className="p-3 rounded-lg bg-zinc-900/60 border border-zinc-800/80">
                        <div className="text-[10px] text-zinc-500 uppercase">MAE</div>
                        <div className="text-lg font-bold text-zinc-200">{predictionResult.metrics?.mae?.toFixed(4)}</div>
                      </div>

                      <div className="p-3 rounded-lg bg-zinc-900/60 border border-zinc-800/80">
                        <div className="text-[10px] text-zinc-500 uppercase">Runtime</div>
                        <div className="text-lg font-bold text-zinc-200">{predictionResult.runtime_ms?.toFixed(1)} ms</div>
                      </div>
                    </div>
                  ) : (
                    <p className="text-xs text-zinc-500 italic">No interpolation executed yet.</p>
                  )}
                </div>

                {/* Export panel */}
                {predictionResult && (
                  <div className="space-y-3 pt-4 border-t border-zinc-800/80">
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-500">Export Findings</h3>
                    
                    <a
                      href={getPredictionUrl()}
                      download={`FrameSat_Prediction_${selectedScene}.png`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="w-full py-2.5 px-4 rounded-lg bg-zinc-900 border border-zinc-800 hover:bg-zinc-800 text-zinc-300 text-xs font-medium flex items-center justify-center gap-2 transition-colors"
                    >
                      <Download className="w-3.5 h-3.5" />
                      Download Prediction Frame
                    </a>
                  </div>
                )}
              </aside>

            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
