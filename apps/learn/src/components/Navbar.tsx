import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { Menu, X, BookOpen, ChevronDown } from "lucide-react";

const chapters = [
  { path: "/data", label: "Data Pipeline", num: "01", ready: true },
  { path: "/tokenizer", label: "Tokenizer", num: "02", ready: true },
  { path: "/pretraining", label: "Pre-training", num: "03", ready: true },
  { path: "/sft", label: "SFT", num: "04", ready: false },
  { path: "/rl", label: "RL Post-training", num: "05", ready: false },
  { path: "/inference", label: "Inference", num: "06", ready: false },
  { path: "/distillation", label: "Distillation", num: "07", ready: false },
];

export function Navbar() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [chaptersOpen, setChaptersOpen] = useState(false);
  const location = useLocation();

  return (
    <nav className="sticky top-0 z-50 border-b border-white/5 bg-surface-900/80 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2.5 group">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-brand-500 to-cyan-500 flex items-center justify-center glow-brand-sm">
              <BookOpen size={16} className="text-white" />
            </div>
            <span className="font-semibold text-white text-sm">
              unbox<span className="text-brand-400">.learn</span>
            </span>
          </Link>

          {/* Desktop nav */}
          <div className="hidden md:flex items-center gap-1">
            <Link
              to="/"
              className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
                location.pathname === "/"
                  ? "text-white bg-white/5"
                  : "text-gray-400 hover:text-white hover:bg-white/5"
              }`}
            >
              Overview
            </Link>

            {/* Chapters dropdown */}
            <div className="relative">
              <button
                onClick={() => setChaptersOpen(!chaptersOpen)}
                className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-sm text-gray-400 hover:text-white hover:bg-white/5 transition-colors"
              >
                Chapters
                <ChevronDown
                  size={14}
                  className={`transition-transform ${chaptersOpen ? "rotate-180" : ""}`}
                />
              </button>
              <AnimatePresence>
                {chaptersOpen && (
                  <motion.div
                    initial={{ opacity: 0, y: -8, scale: 0.96 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -8, scale: 0.96 }}
                    transition={{ duration: 0.15 }}
                    className="absolute top-full mt-1 right-0 w-56 bg-surface-800 border border-white/8 rounded-xl shadow-2xl p-1 overflow-hidden"
                    onMouseLeave={() => setChaptersOpen(false)}
                  >
                    {chapters.map((ch) => (
                      <Link
                        key={ch.path}
                        to={ch.path}
                        onClick={() => setChaptersOpen(false)}
                        className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
                          location.pathname === ch.path
                            ? "bg-brand-500/15 text-brand-300"
                            : "text-gray-400 hover:text-white hover:bg-white/5"
                        } ${!ch.ready ? "opacity-50" : ""}`}
                      >
                        <span className="font-mono text-xs text-gray-600">{ch.num}</span>
                        <span>{ch.label}</span>
                        {!ch.ready && (
                          <span className="ml-auto text-xs px-1.5 py-0.5 rounded bg-surface-600 text-gray-500">
                            Soon
                          </span>
                        )}
                      </Link>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            <a
              href="https://github.com/tianhaoz95/unbox.ai"
              target="_blank"
              rel="noopener noreferrer"
              className="ml-2 px-3 py-1.5 rounded-lg text-sm text-gray-400 hover:text-white hover:bg-white/5 transition-colors"
            >
              GitHub
            </a>
          </div>

          {/* Mobile toggle */}
          <button
            className="md:hidden p-2 rounded-lg text-gray-400 hover:text-white"
            onClick={() => setMobileOpen(!mobileOpen)}
          >
            {mobileOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="md:hidden border-t border-white/5 bg-surface-800 overflow-hidden"
          >
            <div className="px-4 py-4 space-y-1">
              <Link
                to="/"
                onClick={() => setMobileOpen(false)}
                className="block px-3 py-2 rounded-lg text-sm text-gray-300 hover:text-white hover:bg-white/5"
              >
                Overview
              </Link>
              {chapters.map((ch) => (
                <Link
                  key={ch.path}
                  to={ch.path}
                  onClick={() => setMobileOpen(false)}
                  className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm ${
                    location.pathname === ch.path ? "text-brand-300 bg-brand-500/10" : "text-gray-400 hover:text-white hover:bg-white/5"
                  } ${!ch.ready ? "opacity-50" : ""}`}
                >
                  <span className="font-mono text-xs text-gray-600">{ch.num}</span>
                  {ch.label}
                  {!ch.ready && <span className="ml-auto text-xs text-gray-600">Soon</span>}
                </Link>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
}
