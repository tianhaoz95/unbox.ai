import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

const checkpoints = [
  { step: 0, loss: 10.82, ppl: Math.round(Math.exp(10.82)) },
  { step: 500, loss: 5.89, ppl: Math.round(Math.exp(5.89)) },
  { step: 1000, loss: 4.23, ppl: Math.round(Math.exp(4.23)) },
  { step: 3000, loss: 3.41, ppl: Math.round(Math.exp(3.41)) },
  { step: 7000, loss: 2.98, ppl: Math.round(Math.exp(2.98)) },
  { step: 10000, loss: 2.87, ppl: Math.round(Math.exp(2.87)) },
];

const references = [
  { model: "Random (32k vocab)", ppl: 32000, color: "#ef4444" },
  { model: "5-gram LM", ppl: 180, color: "#f59e0b" },
  { model: "GPT-2 (1.5B)", ppl: 35, color: "#a855f7" },
  { model: "Our model @10k steps", ppl: 18, color: "#0ea5e9" },
  { model: "LLaMA 2 7B", ppl: 9, color: "#10b981" },
];

export function PerplexityAnim() {
  const [idx, setIdx] = useState(0);
  const [showRef, setShowRef] = useState(false);

  useEffect(() => {
    const t = setInterval(() => {
      setIdx((i) => {
        if (i < checkpoints.length - 1) return i + 1;
        setShowRef(true);
        clearInterval(t);
        return i;
      });
    }, 800);
    return () => clearInterval(t);
  }, []);

  const current = checkpoints[idx];

  return (
    <div className="card-glass p-6">
      <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">
        Perplexity over training
      </div>

      {/* PPL meter */}
      <div className="flex items-end gap-4 mb-5">
        <div>
          <div className="text-xs text-gray-600 mb-1">Step</div>
          <div className="font-mono text-2xl font-bold text-white">
            {current.step.toLocaleString()}
          </div>
        </div>
        <div className="text-gray-600 text-2xl mb-1">→</div>
        <div>
          <div className="text-xs text-gray-600 mb-1">Loss</div>
          <div className="font-mono text-2xl font-bold text-amber-400">
            {current.loss.toFixed(2)}
          </div>
        </div>
        <div className="text-gray-600 text-2xl mb-1">→</div>
        <div>
          <div className="text-xs text-gray-600 mb-1">Perplexity = e^loss</div>
          <AnimatePresence mode="wait">
            <motion.div
              key={idx}
              initial={{ scale: 1.3, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="font-mono text-2xl font-bold text-brand-300"
            >
              {current.ppl.toLocaleString()}
            </motion.div>
          </AnimatePresence>
        </div>
      </div>

      {/* Bar chart */}
      <div className="space-y-1.5 mb-4">
        {checkpoints.map((ck, i) => (
          <div key={ck.step} className="flex items-center gap-2">
            <span className="font-mono text-xs text-gray-600 w-10 text-right">
              {ck.step === 0 ? "init" : `${(ck.step / 1000).toFixed(0)}k`}
            </span>
            <div className="flex-1 h-4 bg-surface-700 rounded overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: i <= idx ? `${(ck.loss / 11) * 100}%` : 0 }}
                transition={{ delay: i * 0.1 }}
                className="h-full rounded"
                style={{
                  background: i === idx
                    ? "#0ea5e9"
                    : i < idx ? "#1e40af" : "#1e293b",
                }}
              />
            </div>
            <span className="font-mono text-xs text-gray-500 w-8">
              {i <= idx ? ck.loss.toFixed(1) : ""}
            </span>
          </div>
        ))}
      </div>

      {/* Reference comparison — always rendered, opacity-only fade to prevent layout shifts */}
      <motion.div
        animate={{ opacity: showRef ? 1 : 0 }}
        transition={{ duration: 0.4 }}
      >
        <div className="text-xs text-gray-600 mb-2">PPL benchmarks</div>
        <div className="space-y-1">
          {references.map((ref) => (
            <div key={ref.model} className="flex items-center gap-2">
              <div
                className="w-2 h-2 rounded-full flex-shrink-0"
                style={{ background: ref.color }}
              />
              <span className="text-xs text-gray-400 flex-1">{ref.model}</span>
              <span
                className="font-mono text-xs font-semibold"
                style={{ color: ref.color }}
              >
                {ref.ppl.toLocaleString()}
              </span>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
