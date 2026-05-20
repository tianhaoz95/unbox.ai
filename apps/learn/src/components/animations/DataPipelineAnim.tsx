import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

const stages = [
  { id: "raw", label: "Raw Web", icon: "🌐", color: "#6366f1", desc: "CommonCrawl, books, code" },
  { id: "filter", label: "Quality Filter", icon: "🔍", color: "#f59e0b", desc: "Dedup, language, quality" },
  { id: "tokenize", label: "Tokenize", icon: "✂️", color: "#10b981", desc: "BPE → token IDs" },
  { id: "pack", label: "Pack & Batch", icon: "📦", color: "#0ea5e9", desc: "Sequence packing + chunks" },
  { id: "train", label: "Training", icon: "⚡", color: "#a855f7", desc: "Feed to model" },
];

const sampleDocs = [
  "The quick brown fox jumps over the lazy dog.",
  "In 2024, large language models...",
  "def fibonacci(n): return n if n <= 1...",
  "Machine learning is a subset of AI...",
];

export function DataPipelineAnim() {
  const [activeStage, setActiveStage] = useState(0);
  const [_packets, setPackets] = useState<number[]>([]);
  const [currentDoc, setCurrentDoc] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveStage((s) => (s + 1) % stages.length);
      setCurrentDoc((d) => (d + 1) % sampleDocs.length);
    }, 2200);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const id = setTimeout(() => {
      setPackets((p) => [...p.slice(-5), Date.now()]);
    }, 300);
    return () => clearTimeout(id);
  }, [activeStage]);

  return (
    <div className="card-glass p-6">
      <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">
        Pipeline Visualization
      </div>

      {/* Stage flow — py-5 gives 20px clearance so the 16px glow shadow fits
          inside the overflow-x-auto clipping region without being cut off */}
      <div className="flex items-center gap-1 mb-6 overflow-x-auto py-5">
        {stages.map((stage, i) => (
          <div key={stage.id} className="flex items-center gap-1 flex-shrink-0">
            <div
              className="flex flex-col items-center gap-1.5 cursor-pointer"
              onClick={() => setActiveStage(i)}
            >
              {/* Shadow lives on the icon div so it stays well within the
                  overflow container; "none" → shadow can't be interpolated,
                  so inactive state uses a zero-blur transparent value */}
              <motion.div
                animate={{
                  scale: activeStage === i ? 1.05 : 1,
                  boxShadow: activeStage === i
                    ? `0 0 16px ${stage.color}cc`
                    : `0 0 0px ${stage.color}00`,
                }}
                transition={{ duration: 0.3 }}
                className="w-12 h-12 rounded-xl flex items-center justify-center text-xl"
                style={{
                  background: activeStage === i
                    ? `${stage.color}25`
                    : "rgba(255,255,255,0.04)",
                  border: `1px solid ${activeStage === i ? stage.color + "60" : "rgba(255,255,255,0.08)"}`,
                }}
              >
                {stage.icon}
              </motion.div>
              <span
                className="text-xs font-medium whitespace-nowrap transition-colors"
                style={{ color: activeStage === i ? stage.color : "#6b7280" }}
              >
                {stage.label}
              </span>
            </div>
            {i < stages.length - 1 && (
              <div className="w-6 flex items-center justify-center -mt-4">
                <motion.div
                  animate={{ opacity: [0.3, 1, 0.3] }}
                  transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.4 }}
                  className="text-gray-600 text-sm"
                >
                  →
                </motion.div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Active stage detail — fixed height to prevent layout shifts */}
      <div className="h-[110px] overflow-hidden">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeStage}
            initial={{ opacity: 0, x: 10 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -10 }}
            transition={{ duration: 0.25 }}
            className="rounded-xl p-4 border h-full"
            style={{
              background: `${stages[activeStage].color}0d`,
              borderColor: `${stages[activeStage].color}30`,
            }}
          >
            <div className="flex items-start gap-3">
              <span className="text-2xl">{stages[activeStage].icon}</span>
              <div className="flex-1 min-w-0">
                <div
                  className="font-semibold text-sm mb-1"
                  style={{ color: stages[activeStage].color }}
                >
                  {stages[activeStage].label}
                </div>
                <div className="text-xs text-gray-400 mb-2">{stages[activeStage].desc}</div>

                {/* Animated document snippet */}
                {activeStage === 0 && (
                  <AnimatePresence mode="wait">
                    <motion.div
                      key={currentDoc}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="font-mono text-xs text-gray-400 bg-black/20 rounded p-2 truncate"
                    >
                      "{sampleDocs[currentDoc]}"
                    </motion.div>
                  </AnimatePresence>
                )}

                {activeStage === 2 && (
                  <div className="flex flex-wrap gap-1 mt-1">
                    {["The", "▁quick", "▁brown", "▁fox", "▁jumps"].map((tok, i) => (
                      <motion.span
                        key={tok}
                        initial={{ scale: 0, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ delay: i * 0.1 }}
                        className="px-1.5 py-0.5 rounded text-xs font-mono"
                        style={{ background: `${stages[activeStage].color}25`, color: stages[activeStage].color }}
                      >
                        {tok}
                      </motion.span>
                    ))}
                  </div>
                )}

                {activeStage === 3 && (
                  <div className="font-mono text-xs text-gray-400 bg-black/20 rounded p-2">
                    <span className="text-emerald-400">[1024 tokens]</span> × <span className="text-brand-400">batch=32</span>
                    <br />
                    <span className="text-gray-600">no padding, full utilization</span>
                  </div>
                )}

                {activeStage === 4 && (
                  <motion.div
                    animate={{ opacity: [0.5, 1, 0.5] }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                    className="font-mono text-xs text-purple-400"
                  >
                    loss: 3.421 → 2.876 → 2.341...
                  </motion.div>
                )}
              </div>
            </div>
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
