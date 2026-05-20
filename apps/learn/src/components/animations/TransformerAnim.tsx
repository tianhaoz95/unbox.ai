import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

const tokens = ["The", "cat", "sat", "on", "the", "mat"];

export function TransformerAnim() {
  const [step, setStep] = useState(0);
  const [predicting, setPredicting] = useState(false);
  const [attentionWeights, setAttentionWeights] = useState<number[][]>([]);

  useEffect(() => {
    // Generate random-ish attention weights
    const weights = tokens.map((_, i) =>
      tokens.map((_, j) => {
        const base = i === j ? 0.35 : 0.1;
        const proximity = Math.exp(-Math.abs(i - j) * 0.5) * 0.2;
        return Math.max(0.02, Math.min(0.9, base + proximity + Math.random() * 0.15));
      })
    );
    setAttentionWeights(weights);
  }, []);

  useEffect(() => {
    const timer = setInterval(() => {
      setStep((s) => {
        const next = (s + 1) % (tokens.length + 2);
        if (next === tokens.length) setPredicting(true);
        else setPredicting(false);
        return next;
      });
    }, 900);
    return () => clearInterval(timer);
  }, []);

  const activeToken = Math.min(step, tokens.length - 1);

  return (
    <div className="card-glass p-6">
      <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">
        Transformer Forward Pass
      </div>

      {/* Token row */}
      <div className="flex gap-1.5 mb-6 flex-wrap">
        {tokens.map((tok, i) => (
          <motion.div
            key={tok}
            animate={{
              scale: i === activeToken ? 1.1 : 1,
              backgroundColor: i === activeToken
                ? "rgba(14, 165, 233, 0.25)"
                : i < activeToken
                ? "rgba(255,255,255,0.06)"
                : "rgba(255,255,255,0.03)",
            }}
            className="px-2.5 py-1.5 rounded-lg font-mono text-sm border border-white/8 text-gray-300 cursor-default"
          >
            {tok}
          </motion.div>
        ))}
        <AnimatePresence>
          {predicting && (
            <motion.div
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0, opacity: 0 }}
              className="px-2.5 py-1.5 rounded-lg font-mono text-sm border border-brand-500/50 bg-brand-500/15 text-brand-300"
            >
              ???
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Attention heatmap */}
      {attentionWeights.length > 0 && (
        <div className="mb-4">
          <div className="text-xs text-gray-600 mb-2">Attention weights (token {activeToken + 1})</div>
          <div className="flex gap-1">
            {attentionWeights[activeToken]?.map((w, j) => (
              <div key={j} className="flex flex-col items-center gap-1">
                <div
                  className="w-8 h-8 rounded flex items-center justify-center text-xs font-mono"
                  style={{
                    backgroundColor: `rgba(14, 165, 233, ${w})`,
                    color: w > 0.4 ? "#fff" : "#6b7280",
                  }}
                >
                  {w.toFixed(1)}
                </div>
                <span className="text-xs text-gray-600">{tokens[j][0]}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Layer indicator */}
      <div className="flex items-center gap-2 mt-4">
        <span className="text-xs text-gray-600">Layer depth</span>
        <div className="flex-1 flex gap-0.5">
          {Array.from({ length: 12 }, (_, i) => (
            <motion.div
              key={i}
              animate={{
                backgroundColor: i <= (step % 12)
                  ? `rgba(14, 165, 233, ${0.3 + (i / 12) * 0.5})`
                  : "rgba(255,255,255,0.05)",
              }}
              className="flex-1 h-1.5 rounded-full"
            />
          ))}
        </div>
        <span className="text-xs font-mono text-brand-400">12L</span>
      </div>

      {/* Loss */}
      <div className="mt-4 flex items-center justify-between px-3 py-2 rounded-lg bg-surface-700/50 border border-white/5">
        <span className="text-xs text-gray-500">Next token loss</span>
        <motion.span
          animate={{ opacity: [1, 0.5, 1] }}
          transition={{ duration: 1.5, repeat: Infinity }}
          className="font-mono text-xs text-amber-400"
        >
          {predicting ? "computing..." : "2.34 nats"}
        </motion.span>
      </div>
    </div>
  );
}
