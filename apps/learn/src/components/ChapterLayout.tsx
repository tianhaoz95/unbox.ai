import { type ReactNode } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowLeft, ArrowRight } from "lucide-react";

interface ChapterLayoutProps {
  num: string;
  title: string;
  subtitle: string;
  color: string;
  children: ReactNode;
  prev?: { path: string; label: string };
  next?: { path: string; label: string };
}

export function ChapterLayout({
  num,
  title,
  subtitle,
  color,
  children,
  prev,
  next,
}: ChapterLayoutProps) {
  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-12">
      {/* Chapter header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="mb-12"
      >
        <div className={`chapter-tag mb-4 ${color} text-white/80 bg-white/5`}>
          Chapter {num}
        </div>
        <h1 className="text-4xl sm:text-5xl font-bold text-white mb-4 leading-tight">
          {title}
        </h1>
        <p className="text-xl text-gray-400 leading-relaxed max-w-2xl">{subtitle}</p>
      </motion.div>

      {/* Table of contents spacer */}
      <div className="h-px bg-gradient-to-r from-white/10 via-white/5 to-transparent mb-12" />

      {/* Content */}
      <div className="space-y-16">{children}</div>

      {/* Chapter navigation */}
      <div className="mt-20 pt-8 border-t border-white/5 flex items-center justify-between">
        {prev ? (
          <Link
            to={prev.path}
            className="flex items-center gap-2 text-sm text-gray-400 hover:text-white transition-colors group"
          >
            <ArrowLeft size={16} className="group-hover:-translate-x-1 transition-transform" />
            <div>
              <div className="text-xs text-gray-600 mb-0.5">Previous</div>
              <div>{prev.label}</div>
            </div>
          </Link>
        ) : (
          <div />
        )}
        {next ? (
          <Link
            to={next.path}
            className="flex items-center gap-2 text-sm text-gray-400 hover:text-white transition-colors group text-right"
          >
            <div>
              <div className="text-xs text-gray-600 mb-0.5">Next Chapter</div>
              <div>{next.label}</div>
            </div>
            <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />
          </Link>
        ) : (
          <div />
        )}
      </div>
    </div>
  );
}
