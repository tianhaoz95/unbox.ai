import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { Menu, X, BookOpen, ChevronDown, LogIn, LogOut } from "lucide-react";
import { useAuth } from "../contexts/AuthContext";
import { AuthModal } from "./AuthModal";

const chapters = [
  { path: "/data", label: "Data Pipeline", num: "01", ready: true },
  { path: "/tokenizer", label: "Tokenizer", num: "02", ready: true },
  { path: "/pretraining", label: "Pre-training", num: "03", ready: true },
  { path: "/eval", label: "Evaluation", num: "04", ready: true },
  { path: "/sft", label: "SFT", num: "05", ready: true },
  { path: "/dpo", label: "DPO", num: "06", ready: true },
  { path: "/inference", label: "Inference", num: "07", ready: false },
  { path: "/distillation", label: "Distillation", num: "08", ready: false },
];

export function Navbar() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [chaptersOpen, setChaptersOpen] = useState(false);
  const [authOpen, setAuthOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const location = useLocation();
  const { user, signOut } = useAuth();

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

            {/* Auth */}
            {user ? (
              <div className="relative ml-1">
                <button
                  onClick={() => setUserMenuOpen((o) => !o)}
                  className="flex items-center gap-2 pl-2 pr-3 py-1.5 rounded-xl hover:bg-white/5 transition-colors"
                >
                  <img
                    src={user.photoURL ?? ""}
                    alt=""
                    className="w-6 h-6 rounded-full"
                  />
                  <span className="text-sm text-gray-300 max-w-[80px] truncate">
                    {user.displayName?.split(" ")[0]}
                  </span>
                </button>
                <AnimatePresence>
                  {userMenuOpen && (
                    <motion.div
                      initial={{ opacity: 0, y: -6, scale: 0.96 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: -6, scale: 0.96 }}
                      transition={{ duration: 0.13 }}
                      className="absolute right-0 top-full mt-1 w-44 bg-surface-800 border border-white/8 rounded-xl shadow-2xl p-1"
                      onMouseLeave={() => setUserMenuOpen(false)}
                    >
                      <div className="px-3 py-2 border-b border-white/5 mb-1">
                        <p className="text-xs text-gray-400 truncate">{user.email}</p>
                      </div>
                      <button
                        onClick={() => { signOut(); setUserMenuOpen(false); }}
                        className="flex items-center gap-2 w-full px-3 py-2 rounded-lg text-sm text-gray-400 hover:text-white hover:bg-white/5 transition-colors"
                      >
                        <LogOut size={14} />
                        Sign out
                      </button>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            ) : (
              <button
                onClick={() => setAuthOpen(true)}
                className="ml-1 flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm text-gray-400 hover:text-white hover:bg-white/5 transition-colors"
              >
                <LogIn size={14} />
                Sign in
              </button>
            )}
          </div>

          <AuthModal open={authOpen} onClose={() => setAuthOpen(false)} />

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
