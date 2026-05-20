import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

const steps = [
  { id: "forward", label: "Forward Pass", icon: "→", color: "#0ea5e9", desc: "Input tokens → logits" },
  { id: "loss", label: "Compute Loss", icon: "Δ", color: "#f59e0b", desc: "Cross-entropy on next token" },
  { id: "backward", label: "Backward Pass", icon: "←", color: "#a855f7", desc: "Gradient backpropagation" },
  { id: "clip", label: "Grad Clip", icon: "✂", color: "#ec4899", desc: "Max norm = 1.0" },
  { id: "step", label: "Optimizer Step", icon: "⚡", color: "#10b981", desc: "AdamW update weights" },
  { id: "sched", label: "LR Schedule", icon: "📉", color: "#6366f1", desc: "Cosine decay + warmup" },
];

function generateLoss(step: number) {
  return Math.max(0.8, 3.5 * Math.exp(-step * 0.003) + 0.5 + (Math.random() - 0.5) * 0.1);
}

export function TrainingLoopAnim() {
  const [activeStep, setActiveStep] = useState(0);
  const [globalStep, setGlobalStep] = useState(0);
  const [lossHistory, setLossHistory] = useState<number[]>([3.5]);
  const [lr, setLr] = useState(0.0);

  useEffect(() => {
    const timer = setInterval(() => {
      setActiveStep((s) => {
        const next = (s + 1) % steps.length;
        if (next === 0) {
          setGlobalStep((g) => {
            const newStep = g + 1;
            setLossHistory((h) => [...h.slice(-19), generateLoss(newStep)]);
            // Warmup then cosine decay
            const maxLr = 3e-4;
            const warmupSteps = 10;
            const newLr = newStep < warmupSteps
              ? (newStep / warmupSteps) * maxLr
              : maxLr * 0.5 * (1 + Math.cos(Math.PI * (newStep - warmupSteps) / 100));
            setLr(newLr);
            return newStep;
          });
        }
        return next;
      });
    }, 700);
    return () => clearInterval(timer);
  }, []);

  const currentLoss = lossHistory[lossHistory.length - 1];
  const maxLoss = Math.max(...lossHistory);
  const minLoss = Math.min(...lossHistory);

  return (
    <div className="card-glass p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
          Training Loop
        </div>
        <div className="font-mono text-xs text-gray-500">
          step <span className="text-brand-400">{globalStep.toLocaleString()}</span>
        </div>
      </div>

      {/* Step cycle */}
      <div className="grid grid-cols-3 gap-2 mb-5">
        {steps.map((step, i) => (
          <motion.div
            key={step.id}
            animate={{
              scale: activeStep === i ? 1.02 : 1,
              opacity: activeStep === i ? 1 : 0.5,
            }}
            className="p-2.5 rounded-lg border text-center"
            style={{
              borderColor: activeStep === i ? `${step.color}50` : "rgba(255,255,255,0.06)",
              background: activeStep === i ? `${step.color}12` : "transparent",
            }}
          >
            <div
              className="text-lg mb-0.5"
              style={{ color: activeStep === i ? step.color : "#4b5563" }}
            >
              {step.icon}
            </div>
            <div
              className="text-xs font-medium"
              style={{ color: activeStep === i ? step.color : "#6b7280" }}
            >
              {step.label}
            </div>
            <AnimatePresence>
              {activeStep === i && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="text-xs text-gray-500 mt-0.5"
                >
                  {step.desc}
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        ))}
      </div>

      {/* Loss chart */}
      <div className="mb-3">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-xs text-gray-600">Training loss</span>
          <span className="font-mono text-xs text-amber-400">{currentLoss.toFixed(3)}</span>
        </div>
        <div className="h-16 bg-surface-700/50 rounded-lg overflow-hidden p-2 relative">
          <svg width="100%" height="100%" viewBox="0 0 200 48" preserveAspectRatio="none">
            <polyline
              points={lossHistory.map((l, i) => {
                const x = (i / Math.max(lossHistory.length - 1, 1)) * 200;
                const y = 48 - ((l - minLoss) / Math.max(maxLoss - minLoss, 0.1)) * 40 - 4;
                return `${x},${y}`;
              }).join(" ")}
              fill="none"
              stroke="#f59e0b"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            {/* Area fill */}
            <polyline
              points={[
                ...lossHistory.map((l, i) => {
                  const x = (i / Math.max(lossHistory.length - 1, 1)) * 200;
                  const y = 48 - ((l - minLoss) / Math.max(maxLoss - minLoss, 0.1)) * 40 - 4;
                  return `${x},${y}`;
                }),
                `200,48`, `0,48`
              ].join(" ")}
              fill="rgba(245, 158, 11, 0.06)"
            />
          </svg>
        </div>
      </div>

      {/* LR meter */}
      <div className="flex items-center justify-between">
        <span className="text-xs text-gray-600">Learning rate</span>
        <div className="flex items-center gap-2">
          <div className="w-24 h-1 bg-surface-700 rounded-full overflow-hidden">
            <motion.div
              animate={{ width: `${(lr / 3e-4) * 100}%` }}
              className="h-full bg-gradient-to-r from-brand-600 to-cyan-400 rounded-full"
            />
          </div>
          <span className="font-mono text-xs text-brand-400">{lr.toExponential(1)}</span>
        </div>
      </div>
    </div>
  );
}
