import { type ReactNode } from "react";
import { Lightbulb, AlertTriangle, Info, Zap } from "lucide-react";

type CalloutType = "why" | "tip" | "warning" | "insight";

interface CalloutProps {
  type: CalloutType;
  title?: string;
  children: ReactNode;
}

const config = {
  why: {
    icon: Lightbulb,
    color: "text-amber-400",
    bg: "bg-amber-500/8",
    border: "border-amber-500/20",
    label: "Why this matters",
  },
  tip: {
    icon: Zap,
    color: "text-brand-400",
    bg: "bg-brand-500/8",
    border: "border-brand-500/20",
    label: "Pro tip",
  },
  warning: {
    icon: AlertTriangle,
    color: "text-orange-400",
    bg: "bg-orange-500/8",
    border: "border-orange-500/20",
    label: "Watch out",
  },
  insight: {
    icon: Info,
    color: "text-emerald-400",
    bg: "bg-emerald-500/8",
    border: "border-emerald-500/20",
    label: "Key insight",
  },
};

export function Callout({ type, title, children }: CalloutProps) {
  const { icon: Icon, color, bg, border, label } = config[type];

  return (
    <div className={`rounded-xl border ${bg} ${border} p-4 flex gap-3`}>
      <div className={`mt-0.5 flex-shrink-0 ${color}`}>
        <Icon size={16} />
      </div>
      <div>
        <p className={`text-xs font-semibold uppercase tracking-wider ${color} mb-1.5`}>
          {title || label}
        </p>
        <div className="text-sm text-gray-300 leading-relaxed prose-custom">
          {children}
        </div>
      </div>
    </div>
  );
}
