import { useState } from "react";
import { type ReactNode } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowLeft, ArrowRight, MessageSquare } from "lucide-react";
import { useLocation } from "react-router-dom";
import { SelectionCommentLayer } from "./SelectionCommentLayer";
import { CommentsPanel } from "./CommentsPanel";
import { AuthModal } from "./AuthModal";
import { useComments } from "../hooks/useComments";
import { useLanguage } from "../contexts/LanguageContext";

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
  const { pathname } = useLocation();
  const { comments, loading, addComment } = useComments(pathname);
  const [panelOpen, setPanelOpen] = useState(false);
  const [authOpen, setAuthOpen] = useState(false);
  const { t } = useLanguage();

  const commentLabel =
    comments.length > 0
      ? comments.length === 1
        ? t("comments.count").replace("{n}", String(comments.length))
        : t("comments.countMany").replace("{n}", String(comments.length))
      : t("chapter.comments");

  return (
    <>
      {/* Floating comments button */}
      <motion.button
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.6 }}
        onClick={() => setPanelOpen(true)}
        className="fixed bottom-6 right-6 z-30 flex items-center gap-2 px-4 py-2.5 rounded-2xl bg-surface-700 border border-white/10 text-sm text-gray-300 hover:text-white hover:bg-surface-600 shadow-xl transition-colors"
      >
        <MessageSquare size={16} className="text-brand-400" />
        {comments.length > 0 ? (
          <span>
            <span className="text-brand-300 font-semibold">{comments.length}</span>{" "}
            {comments.length === 1
              ? t("comments.count").replace("{n}", "").trim()
              : t("comments.countMany").replace("{n}", "").trim()}
          </span>
        ) : (
          commentLabel
        )}
      </motion.button>

      <CommentsPanel
        open={panelOpen}
        onClose={() => setPanelOpen(false)}
        comments={comments}
        loading={loading}
        chapterTitle={`Chapter ${num} — ${title}`}
      />

      <AuthModal open={authOpen} onClose={() => setAuthOpen(false)} />

      <SelectionCommentLayer
        comments={comments}
        onAddComment={addComment}
        onRequestSignIn={() => setAuthOpen(true)}
      >
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

          {/* Divider */}
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
                  <div className="text-xs text-gray-600 mb-0.5">{t("chapter.prev")}</div>
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
                  <div className="text-xs text-gray-600 mb-0.5">{t("chapter.next")}</div>
                  <div>{next.label}</div>
                </div>
                <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />
              </Link>
            ) : (
              <div />
            )}
          </div>
        </div>
      </SelectionCommentLayer>
    </>
  );
}
