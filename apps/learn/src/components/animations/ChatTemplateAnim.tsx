import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

type Role = "system" | "user" | "assistant";

const conversation = [
  { role: "system" as Role, content: "You are a helpful assistant." },
  { role: "user" as Role, content: "What is backpropagation?" },
  {
    role: "assistant" as Role,
    content:
      "Backpropagation is the algorithm used to compute gradients in neural networks by applying the chain rule backwards through the computation graph.",
  },
];

const roleConfig = {
  system: { color: "#6366f1", label: "system", bg: "rgba(99,102,241,0.1)", border: "rgba(99,102,241,0.3)" },
  user: { color: "#0ea5e9", label: "user", bg: "rgba(14,165,233,0.1)", border: "rgba(14,165,233,0.3)" },
  assistant: { color: "#10b981", label: "assistant", bg: "rgba(16,185,129,0.1)", border: "rgba(16,185,129,0.3)" },
};

const templateSteps = [
  { label: "Messages (Python list)", view: "messages" },
  { label: "After apply_chat_template()", view: "template" },
  { label: "Token IDs (with masking)", view: "tokens" },
];

export function ChatTemplateAnim() {
  const [step, setStep] = useState(0);

  useEffect(() => {
    const t = setInterval(() => setStep((s) => (s + 1) % templateSteps.length), 2800);
    return () => clearInterval(t);
  }, []);

  const templateText =
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" +
    "<|im_start|>user\nWhat is backpropagation?<|im_end|>\n" +
    "<|im_start|>assistant\nBackpropagation is the algorithm...<|im_end|>";

  const tokenRows = [
    { text: "<|im_start|>system\\nYou are...<|im_end|>\\n<|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n", masked: true, label: "masked (prompt)" },
    { text: "Backpropagation is the algorithm used to compute...", masked: false, label: "loss computed here" },
    { text: "<|im_end|>", masked: false, label: "stop token" },
  ];

  return (
    <div className="card-glass p-6">
      <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">
        Chat Template Formatting
      </div>

      {/* Step tabs */}
      <div className="flex gap-1 mb-4 p-1 bg-surface-700/50 rounded-xl">
        {templateSteps.map((s, i) => (
          <button
            key={i}
            onClick={() => setStep(i)}
            className="flex-1 text-xs py-1.5 px-2 rounded-lg transition-all"
            style={{
              background: step === i ? "rgba(14,165,233,0.2)" : "transparent",
              color: step === i ? "#38bdf8" : "#6b7280",
            }}
          >
            {i + 1}. {s.label.split("(")[0].trim()}
          </button>
        ))}
      </div>

      {/* Fixed height content area to prevent layout shifts between views */}
      <div className="h-[220px] overflow-hidden relative">
        <AnimatePresence mode="wait">
          {step === 0 && (
            <motion.div key="messages" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="absolute inset-0 overflow-hidden">
              <div className="space-y-2">
                {conversation.map((msg, i) => {
                  const cfg = roleConfig[msg.role];
                  return (
                    <motion.div
                      key={i}
                      initial={{ x: -10, opacity: 0 }}
                      animate={{ x: 0, opacity: 1 }}
                      transition={{ delay: i * 0.12 }}
                      className="rounded-lg p-3 border"
                      style={{ background: cfg.bg, borderColor: cfg.border }}
                    >
                      <span className="text-xs font-mono font-semibold" style={{ color: cfg.color }}>
                        {cfg.label}
                      </span>
                      <p className="text-xs text-gray-300 mt-1 leading-relaxed">{msg.content}</p>
                    </motion.div>
                  );
                })}
              </div>
            </motion.div>
          )}

          {step === 1 && (
            <motion.div key="template" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="absolute inset-0 overflow-hidden">
              <div className="font-mono text-xs bg-surface-700/50 rounded-lg p-3 text-gray-300 leading-loose whitespace-pre-wrap break-all h-full overflow-hidden">
                {templateText.split("\n").map((line, i) => (
                  <span key={i}>
                    {line.includes("<|im_start|>system") ? (
                      <span className="text-indigo-400">{line}</span>
                    ) : line.includes("<|im_start|>user") ? (
                      <span className="text-brand-400">{line}</span>
                    ) : line.includes("<|im_start|>assistant") ? (
                      <span className="text-emerald-400">{line}</span>
                    ) : line.includes("<|im_end|>") ? (
                      <span className="text-gray-500">{line}</span>
                    ) : (
                      line
                    )}
                    {"\n"}
                  </span>
                ))}
              </div>
            </motion.div>
          )}

          {step === 2 && (
            <motion.div key="tokens" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="absolute inset-0 overflow-hidden">
              <div className="space-y-2">
                {tokenRows.map((row, i) => (
                  <div
                    key={i}
                    className="rounded-lg p-3 border"
                    style={{
                      background: row.masked ? "rgba(239,68,68,0.06)" : "rgba(16,185,129,0.08)",
                      borderColor: row.masked ? "rgba(239,68,68,0.2)" : "rgba(16,185,129,0.2)",
                    }}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-mono truncate text-gray-400 max-w-[200px]">
                        {row.text}
                      </span>
                      <span
                        className="text-xs font-semibold ml-2 flex-shrink-0"
                        style={{ color: row.masked ? "#ef4444" : "#10b981" }}
                      >
                        {row.label}
                      </span>
                    </div>
                    <div className="flex items-center gap-1 mt-1">
                      {row.masked ? (
                        <span className="text-xs text-red-400">labels = -100 (ignored by cross-entropy)</span>
                      ) : (
                        <span className="text-xs text-emerald-400">loss computed → model learns response</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
              <p className="text-xs text-gray-600 mt-3">
                <code className="text-brand-400">completion_only_loss=True</code> masks prompt tokens automatically
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
