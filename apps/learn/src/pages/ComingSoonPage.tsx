import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowLeft, Bell, GitBranch } from "lucide-react";

interface ComingSoonPageProps {
  chapter: string;
}

const chapterDetails: Record<string, { num: string; icon: string; desc: string; topics: string[]; color: string }> = {
  SFT: {
    num: "04",
    icon: "💬",
    color: "#f59e0b",
    desc: "Supervised Fine-Tuning turns a base model into an instruction follower using labeled (prompt, response) pairs.",
    topics: [
      "Chat template formatting (ChatML, Alpaca, Llama-2)",
      "TRL SFTTrainer with SFTConfig",
      "LoRA / QLoRA parameter-efficient fine-tuning",
      "Creating and curating SFT datasets",
      "Overfitting and early stopping",
      "Evaluating instruction following quality",
    ],
  },
  "RL Post-training": {
    num: "05",
    icon: "🏆",
    color: "#ec4899",
    desc: "Reinforce the model with human preferences and verifiable rewards to improve helpfulness and accuracy.",
    topics: [
      "DPO: Direct Preference Optimization",
      "GRPO: Group Relative Policy Optimization",
      "RLVR: Reinforcement from Verifiable Rewards",
      "Reward model training",
      "PPO with OpenRLHF",
      "Dataset formats: chosen/rejected pairs",
    ],
  },
  Inference: {
    num: "06",
    icon: "🚀",
    color: "#a855f7",
    desc: "Build a production-grade inference engine from scratch: continuous batching, paged KV cache, and disaggregated serving.",
    topics: [
      "PagedAttention and paged KV cache",
      "Continuous batching scheduler",
      "Disaggregated prefill-decode architecture",
      "Triton kernels: paged decode attention",
      "ZMQ-based worker communication",
      "OpenAI-compatible FastAPI server",
    ],
  },
  Distillation: {
    num: "07",
    icon: "🔬",
    color: "#14b8a6",
    desc: "Compress a large teacher model's knowledge into a smaller student via logit matching and hidden state alignment.",
    topics: [
      "Logit matching (KL divergence)",
      "Hidden state distillation",
      "TRL GKDTrainer",
      "Reasoning chain distillation",
      "Choosing teacher-student pairs",
      "Evaluation: QA benchmarks",
    ],
  },
};

export function ComingSoonPage({ chapter }: ComingSoonPageProps) {
  const details = chapterDetails[chapter];
  if (!details) return null;

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-16">
      <Link
        to="/"
        className="inline-flex items-center gap-2 text-sm text-gray-500 hover:text-gray-300 transition-colors mb-10"
      >
        <ArrowLeft size={14} />
        Back to overview
      </Link>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        {/* Header */}
        <div className="text-center mb-12">
          <div
            className="w-20 h-20 rounded-2xl flex items-center justify-center text-4xl mx-auto mb-6"
            style={{ background: `${details.color}15`, border: `1px solid ${details.color}30` }}
          >
            {details.icon}
          </div>

          <div className="font-mono text-xs text-gray-600 mb-2">Chapter {details.num}</div>
          <h1 className="text-4xl font-bold text-white mb-4">{chapter}</h1>
          <p className="text-lg text-gray-400 max-w-2xl mx-auto">{details.desc}</p>
        </div>

        {/* Coming soon badge */}
        <div className="text-center mb-10">
          <motion.div
            animate={{ scale: [1, 1.03, 1] }}
            transition={{ duration: 2.5, repeat: Infinity }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-semibold"
            style={{
              background: `${details.color}15`,
              border: `1px solid ${details.color}40`,
              color: details.color,
            }}
          >
            <Bell size={14} />
            Coming soon — in development
          </motion.div>
        </div>

        {/* Topics preview */}
        <div className="card-glass p-6 mb-8">
          <h2 className="text-lg font-bold text-white mb-4">What you'll learn</h2>
          <div className="grid sm:grid-cols-2 gap-2">
            {details.topics.map((topic, i) => (
              <motion.div
                key={topic}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.07 }}
                className="flex items-start gap-2.5 py-2 border-b border-white/3 last:border-0"
              >
                <span
                  className="w-1.5 h-1.5 rounded-full mt-1.5 flex-shrink-0"
                  style={{ background: details.color }}
                />
                <span className="text-sm text-gray-300">{topic}</span>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Nav to available chapters */}
        <div className="text-center">
          <p className="text-sm text-gray-500 mb-4">
            In the meantime, explore the available chapters:
          </p>
          <div className="flex flex-wrap justify-center gap-3">
            {[
              { path: "/data", label: "01 — Data Pipeline", color: "#6366f1" },
              { path: "/tokenizer", label: "02 — Tokenizer", color: "#10b981" },
              { path: "/pretraining", label: "03 — Pre-training", color: "#0ea5e9" },
            ].map((ch) => (
              <Link
                key={ch.path}
                to={ch.path}
                className="px-4 py-2 rounded-xl text-sm font-medium transition-colors border"
                style={{
                  color: ch.color,
                  borderColor: `${ch.color}30`,
                  background: `${ch.color}0a`,
                }}
              >
                {ch.label}
              </Link>
            ))}
          </div>
        </div>

        {/* GitHub link */}
        <div className="mt-12 text-center">
          <a
            href="https://github.com/tianhaoz95/unbox.ai"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 text-sm text-gray-500 hover:text-gray-300 transition-colors"
          >
            <GitBranch size={14} />
            Follow progress on GitHub
          </a>
        </div>
      </motion.div>
    </div>
  );
}
