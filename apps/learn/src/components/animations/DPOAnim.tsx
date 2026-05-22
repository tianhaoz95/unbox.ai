import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

const pairs = [
  {
    prompt: "Explain gradient descent.",
    chosen: "Gradient descent iteratively adjusts weights by moving in the direction that reduces loss, scaled by the learning rate.",
    rejected: "It's a way to train neural nets by adjusting stuff.",
    chosen_lp_before: -2.1,
    rejected_lp_before: -2.4,
    chosen_lp_after: -1.6,
    rejected_lp_after: -3.1,
  },
  {
    prompt: "What is attention in transformers?",
    chosen: "Attention allows each token to selectively weight other tokens when computing its representation, enabling context-aware embeddings.",
    rejected: "Attention means the model pays attention to words.",
    chosen_lp_before: -2.3,
    rejected_lp_before: -2.5,
    chosen_lp_after: -1.8,
    rejected_lp_after: -3.4,
  },
];

type Phase = "before" | "after";

export function DPOAnim() {
  const [pairIdx, setPairIdx] = useState(0);
  const [phase, setPhase] = useState<Phase>("before");

  useEffect(() => {
    const t = setInterval(() => {
      setPhase((p) => {
        if (p === "before") return "after";
        setPairIdx((i) => (i + 1) % pairs.length);
        return "before";
      });
    }, 2500);
    return () => clearInterval(t);
  }, []);

  const pair = pairs[pairIdx];
  const chosenLP = phase === "before" ? pair.chosen_lp_before : pair.chosen_lp_after;
  const rejLP = phase === "before" ? pair.rejected_lp_before : pair.rejected_lp_after;
  const margin = chosenLP - rejLP;
  const wins = chosenLP > rejLP;

  return (
    <div className="card-glass p-6">
      <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">
        DPO: Preference Learning
      </div>
      <div className="text-xs text-gray-600 mb-4">
        Push log P(chosen) up, log P(rejected) down
      </div>

      {/* Phase badge */}
      <div className="flex gap-2 mb-4">
        {(["before", "after"] as Phase[]).map((p) => (
          <button
            key={p}
            onClick={() => setPhase(p)}
            className="px-3 py-1 rounded-full text-xs font-semibold transition-all"
            style={{
              background: phase === p ? (p === "after" ? "rgba(16,185,129,0.2)" : "rgba(14,165,233,0.15)") : "rgba(255,255,255,0.04)",
              color: phase === p ? (p === "after" ? "#10b981" : "#38bdf8") : "#6b7280",
              border: `1px solid ${phase === p ? (p === "after" ? "rgba(16,185,129,0.4)" : "rgba(14,165,233,0.3)") : "rgba(255,255,255,0.06)"}`,
            }}
          >
            {p === "before" ? "SFT baseline" : "After DPO"}
          </button>
        ))}
      </div>

      {/* Prompt */}
      <div className="mb-3 px-3 py-2 bg-surface-700/50 rounded-lg border border-white/5">
        <span className="text-xs text-gray-500">Prompt: </span>
        <span className="text-xs text-gray-300">{pair.prompt}</span>
      </div>

      {/* Chosen vs Rejected */}
      <div className="space-y-2 mb-4">
        {[
          { label: "chosen", text: pair.chosen, lp: chosenLP, color: "#10b981" },
          { label: "rejected", text: pair.rejected, lp: rejLP, color: "#ef4444" },
        ].map((item) => (
          <div
            key={item.label}
            className="rounded-lg p-3 border"
            style={{
              background: `${item.color}08`,
              borderColor: `${item.color}25`,
            }}
          >
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-semibold" style={{ color: item.color }}>
                {item.label}
              </span>
              <AnimatePresence mode="wait">
                <motion.span
                  key={`${pairIdx}-${phase}-${item.label}`}
                  initial={{ scale: 1.2, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  className="font-mono text-xs font-bold"
                  style={{ color: item.color }}
                >
                  log P = {item.lp.toFixed(2)}
                </motion.span>
              </AnimatePresence>
            </div>
            <p className="text-xs text-gray-400 leading-relaxed">{item.text}</p>
          </div>
        ))}
      </div>

      {/* Margin indicator */}
      <div className="flex items-center justify-between px-3 py-2 rounded-lg bg-surface-700/50 border border-white/5">
        <span className="text-xs text-gray-500">Margin (chosen − rejected)</span>
        <AnimatePresence mode="wait">
          <motion.div
            key={`${pairIdx}-${phase}-margin`}
            initial={{ scale: 1.15, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="flex items-center gap-2"
          >
            <span
              className="font-mono text-sm font-bold"
              style={{ color: wins ? "#10b981" : "#ef4444" }}
            >
              {margin > 0 ? "+" : ""}{margin.toFixed(2)}
            </span>
            <span className="text-xs" style={{ color: wins ? "#10b981" : "#ef4444" }}>
              {wins ? "✓ win" : "✗ loss"}
            </span>
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Win rate */}
      <div className="mt-3 flex items-center justify-between">
        <span className="text-xs text-gray-600">Win rate on 500 pairs</span>
        <span
          className="font-mono text-xs font-semibold"
          style={{ color: phase === "after" ? "#10b981" : "#0ea5e9" }}
        >
          {phase === "before" ? "~58%" : "~73%"}
        </span>
      </div>
    </div>
  );
}
