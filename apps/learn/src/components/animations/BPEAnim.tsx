import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

// Simulate BPE merges step by step
const bpeSteps = [
  {
    label: "Initial: character split",
    chars: [["l","o","w"], ["l","o","w","e","r"], ["n","e","w"], ["w","i","d","e","r"]],
    pairs: { "l+o": 2, "o+w": 2, "e+r": 2, "w+e": 1 },
    highlight: "l+o",
  },
  {
    label: "Merge #1: (l, o) → lo",
    chars: [["lo","w"], ["lo","w","e","r"], ["n","e","w"], ["w","i","d","e","r"]],
    pairs: { "lo+w": 2, "o+w": 0, "e+r": 2, "w+e": 1 },
    highlight: "lo+w",
  },
  {
    label: "Merge #2: (lo, w) → low",
    chars: [["low"], ["low","e","r"], ["n","e","w"], ["w","i","d","e","r"]],
    pairs: { "low+e": 1, "e+r": 2, "n+e": 1, "i+d": 1 },
    highlight: "e+r",
  },
  {
    label: "Merge #3: (e, r) → er",
    chars: [["low"], ["low","er"], ["n","e","w"], ["w","i","d","er"]],
    pairs: { "low+er": 1, "n+e": 1, "d+er": 1 },
    highlight: "low+er",
  },
  {
    label: "Result: learned subwords",
    chars: [["low"], ["lower"], ["n","e","w"], ["w","i","d","er"]],
    pairs: {},
    highlight: "",
  },
];

const tokenColors = ["#0ea5e9", "#10b981", "#a855f7", "#f59e0b", "#ec4899", "#6366f1"];

export function BPEAnim() {
  const [step, setStep] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setStep((s) => (s + 1) % bpeSteps.length);
    }, 2500);
    return () => clearInterval(timer);
  }, []);

  const current = bpeSteps[step];

  return (
    <div className="card-glass p-6">
      <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">
        BPE Algorithm
      </div>
      <div className="text-xs text-gray-600 mb-4">
        Iteratively merge the most frequent character pair
      </div>

      {/* Step label */}
      <AnimatePresence mode="wait">
        <motion.div
          key={step}
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          className="text-sm font-medium text-brand-300 mb-4"
        >
          {current.label}
        </motion.div>
      </AnimatePresence>

      {/* Corpus */}
      <div className="space-y-2 mb-5">
        {current.chars.map((word, wi) => (
          <div key={wi} className="flex items-center gap-1.5">
            <span className="text-xs text-gray-600 w-4 text-right">{wi + 1}.</span>
            <div className="flex gap-1">
              {word.map((ch, ci) => (
                <motion.span
                  key={`${step}-${wi}-${ci}`}
                  layout
                  initial={{ scale: 0.85, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  className="px-1.5 py-0.5 rounded text-xs font-mono border"
                  style={{
                    color: tokenColors[ci % tokenColors.length],
                    borderColor: `${tokenColors[ci % tokenColors.length]}40`,
                    background: `${tokenColors[ci % tokenColors.length]}12`,
                  }}
                >
                  {ch}
                </motion.span>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Pair frequencies */}
      {Object.keys(current.pairs).length > 0 && (
        <div>
          <div className="text-xs text-gray-600 mb-2">Pair frequencies</div>
          <div className="space-y-1">
            {Object.entries(current.pairs)
              .filter(([, v]) => v > 0)
              .sort(([, a], [, b]) => b - a)
              .map(([pair, count]) => (
                <div
                  key={pair}
                  className="flex items-center gap-2"
                >
                  <span
                    className="font-mono text-xs px-1.5 py-0.5 rounded"
                    style={{
                      color: pair === current.highlight ? "#0ea5e9" : "#6b7280",
                      background: pair === current.highlight ? "rgba(14,165,233,0.12)" : "rgba(255,255,255,0.04)",
                    }}
                  >
                    {pair}
                  </span>
                  <div className="flex-1 h-1 bg-surface-700 rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${(count / 3) * 100}%` }}
                      className="h-full rounded-full"
                      style={{
                        background: pair === current.highlight ? "#0ea5e9" : "#374151",
                      }}
                    />
                  </div>
                  <span className="text-xs text-gray-600 font-mono">{count}</span>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Progress */}
      <div className="mt-4 flex gap-1.5 justify-center">
        {bpeSteps.map((_, i) => (
          <button
            key={i}
            onClick={() => setStep(i)}
            className={`w-1.5 h-1.5 rounded-full transition-colors ${
              i === step ? "bg-brand-400" : "bg-white/15"
            }`}
          />
        ))}
      </div>
    </div>
  );
}
