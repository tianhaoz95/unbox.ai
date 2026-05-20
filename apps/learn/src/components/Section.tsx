import { type ReactNode } from "react";
import { motion } from "framer-motion";

interface SectionProps {
  stepNum: number;
  title: string;
  children: ReactNode;
}

export function Section({ stepNum, title, children }: SectionProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-80px" }}
      transition={{ duration: 0.5 }}
      className="relative"
    >
      {/* Step marker */}
      <div className="flex items-start gap-4 mb-6">
        <div className="flex-shrink-0 w-9 h-9 rounded-xl bg-brand-500/15 border border-brand-500/30 flex items-center justify-center">
          <span className="font-mono text-xs font-bold text-brand-400">
            {String(stepNum).padStart(2, "0")}
          </span>
        </div>
        <div>
          <h2 className="text-2xl font-bold text-white leading-tight">{title}</h2>
        </div>
      </div>

      {/* Content with left border accent */}
      <div className="ml-4 pl-8 border-l border-white/5 space-y-6">
        {children}
      </div>
    </motion.div>
  );
}
