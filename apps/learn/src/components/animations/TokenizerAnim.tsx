import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

const examples = [
  {
    text: "tokenization",
    tokens: ["token", "ization"],
    ids: [3662, 2065],
    colors: ["#0ea5e9", "#10b981"],
  },
  {
    text: "unbox.ai",
    tokens: ["un", "box", ".", "ai"],
    ids: [521, 3524, 13, 1872],
    colors: ["#a855f7", "#6366f1", "#f59e0b", "#ec4899"],
  },
  {
    text: "Hello world",
    tokens: ["Hello", "▁world"],
    ids: [15496, 995],
    colors: ["#0ea5e9", "#10b981"],
  },
  {
    text: "def forward(x):",
    tokens: ["def", "▁forward", "(", "x", "):"],
    ids: [892, 2651, 7, 87, 2599],
    colors: ["#f59e0b", "#0ea5e9", "#6b7280", "#a855f7", "#ec4899"],
  },
];

export function TokenizerAnim() {
  const [exampleIdx, setExampleIdx] = useState(0);
  const [revealed, setRevealed] = useState(0);
  const [showIds, setShowIds] = useState(false);

  const example = examples[exampleIdx];

  useEffect(() => {
    setRevealed(0);
    setShowIds(false);
    const reveal = setInterval(() => {
      setRevealed((r) => {
        if (r >= example.tokens.length) {
          clearInterval(reveal);
          setTimeout(() => setShowIds(true), 400);
          return r;
        }
        return r + 1;
      });
    }, 350);
    return () => clearInterval(reveal);
  }, [exampleIdx, example.tokens.length]);

  useEffect(() => {
    const timer = setTimeout(() => {
      setExampleIdx((i) => (i + 1) % examples.length);
    }, 4500);
    return () => clearTimeout(timer);
  }, [exampleIdx]);

  return (
    <div className="card-glass p-6">
      <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">
        BPE Tokenization
      </div>

      {/* Input */}
      <div className="mb-4">
        <div className="text-xs text-gray-600 mb-2">Input string</div>
        <div className="font-mono text-sm bg-surface-700/50 rounded-lg px-3 py-2 text-white border border-white/5">
          "{example.text}"
        </div>
      </div>

      {/* Tokens */}
      <div className="mb-4">
        <div className="text-xs text-gray-600 mb-2">Tokens</div>
        <div className="flex flex-wrap gap-2 min-h-[40px]">
          <AnimatePresence>
            {example.tokens.slice(0, revealed).map((tok, i) => (
              <motion.div
                key={`${exampleIdx}-${i}`}
                initial={{ scale: 0.6, opacity: 0, y: 8 }}
                animate={{ scale: 1, opacity: 1, y: 0 }}
                transition={{ type: "spring", stiffness: 300, damping: 20 }}
                className="px-2.5 py-1 rounded-lg font-mono text-sm font-medium"
                style={{
                  background: `${example.colors[i % example.colors.length]}20`,
                  border: `1px solid ${example.colors[i % example.colors.length]}40`,
                  color: example.colors[i % example.colors.length],
                }}
              >
                {tok}
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </div>

      {/* IDs — always reserve space to prevent layout shifts */}
      <div className="h-[52px]">
        <motion.div
          animate={{ opacity: showIds ? 1 : 0 }}
          transition={{ duration: 0.3 }}
        >
          <div className="text-xs text-gray-600 mb-2">Token IDs</div>
          <div className="flex flex-wrap gap-2">
            {example.ids.map((id, i) => (
              <motion.span
                key={`${exampleIdx}-id-${i}`}
                initial={{ opacity: 0 }}
                animate={{ opacity: showIds ? 1 : 0 }}
                transition={{ delay: i * 0.08 }}
                className="font-mono text-xs px-2 py-1 rounded bg-surface-700 text-gray-400 border border-white/5"
              >
                {id}
              </motion.span>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Vocab indicator */}
      <div className="mt-4 pt-4 border-t border-white/5">
        <div className="flex items-center justify-between text-xs text-gray-600">
          <span>Vocabulary size</span>
          <span className="font-mono text-brand-400">32,000 tokens</span>
        </div>
        <div className="mt-1.5 h-1.5 bg-surface-700 rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: "100%" }}
            transition={{ duration: 1.5, ease: "easeOut" }}
            className="h-full bg-gradient-to-r from-brand-600 to-cyan-500 rounded-full"
          />
        </div>
      </div>

      {/* Example selector */}
      <div className="mt-4 flex gap-1.5 justify-center">
        {examples.map((_, i) => (
          <button
            key={i}
            onClick={() => setExampleIdx(i)}
            className={`w-1.5 h-1.5 rounded-full transition-colors ${
              i === exampleIdx ? "bg-brand-400" : "bg-white/20"
            }`}
          />
        ))}
      </div>
    </div>
  );
}
