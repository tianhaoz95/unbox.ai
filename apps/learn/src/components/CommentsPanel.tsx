import { motion, AnimatePresence } from "framer-motion";
import { X, MessageSquare, Clock } from "lucide-react";
import type { Comment } from "../hooks/useComments";

interface Props {
  open: boolean;
  onClose: () => void;
  comments: Comment[];
  loading: boolean;
  chapterTitle: string;
}

function timeAgo(date: Date): string {
  const diff = Math.floor((Date.now() - date.getTime()) / 1000);
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

export function CommentsPanel({ open, onClose, comments, loading, chapterTitle }: Props) {
  return (
    <AnimatePresence>
      {open && (
        <>
          {/* Backdrop */}
          <motion.div
            key="backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/40 z-40"
            onClick={onClose}
          />

          {/* Panel */}
          <motion.div
            key="panel"
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 28, stiffness: 280 }}
            className="fixed right-0 top-0 h-full w-full max-w-sm bg-surface-800 border-l border-white/8 z-50 flex flex-col"
          >
            {/* Header */}
            <div className="flex items-center justify-between px-5 py-4 border-b border-white/8">
              <div>
                <div className="flex items-center gap-2">
                  <MessageSquare size={16} className="text-brand-400" />
                  <span className="font-semibold text-white text-sm">Comments</span>
                  {comments.length > 0 && (
                    <span className="px-1.5 py-0.5 rounded-full text-xs bg-brand-500/20 text-brand-300 font-mono">
                      {comments.length}
                    </span>
                  )}
                </div>
                <p className="text-xs text-gray-500 mt-0.5">{chapterTitle}</p>
              </div>
              <button
                onClick={onClose}
                className="p-1.5 rounded-lg text-gray-500 hover:text-white hover:bg-white/5 transition-colors"
              >
                <X size={16} />
              </button>
            </div>

            {/* Comment list */}
            <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
              {loading ? (
                <div className="space-y-3">
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="rounded-xl bg-surface-700/50 p-4 animate-pulse">
                      <div className="h-3 bg-white/8 rounded w-3/4 mb-2" />
                      <div className="h-2 bg-white/5 rounded w-full mb-1" />
                      <div className="h-2 bg-white/5 rounded w-2/3" />
                    </div>
                  ))}
                </div>
              ) : comments.length === 0 ? (
                <div className="text-center py-16">
                  <MessageSquare size={32} className="text-gray-700 mx-auto mb-3" />
                  <p className="text-sm text-gray-500">No comments yet</p>
                  <p className="text-xs text-gray-600 mt-1">
                    Select any text in the chapter to add one
                  </p>
                </div>
              ) : (
                comments.map((c) => (
                  <motion.div
                    key={c.id}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="rounded-xl bg-surface-700/50 border border-white/5 p-4"
                  >
                    {/* User + time */}
                    <div className="flex items-center gap-2 mb-2">
                      {c.userPhotoURL ? (
                        <img
                          src={c.userPhotoURL}
                          alt=""
                          className="w-6 h-6 rounded-full flex-shrink-0"
                        />
                      ) : (
                        <div className="w-6 h-6 rounded-full bg-brand-500/30 flex items-center justify-center text-xs text-brand-400 flex-shrink-0">
                          {c.userDisplayName[0]}
                        </div>
                      )}
                      <span className="text-xs font-medium text-gray-300 flex-1 truncate">
                        {c.userDisplayName}
                      </span>
                      <span className="flex items-center gap-1 text-xs text-gray-600">
                        <Clock size={10} />
                        {timeAgo(c.createdAt)}
                      </span>
                    </div>

                    {/* Quoted text */}
                    <div className="mb-2 px-2.5 py-1.5 rounded-lg bg-brand-500/8 border border-brand-500/15">
                      <p className="text-xs text-brand-300/80 italic line-clamp-2">
                        "{c.selectedText}"
                      </p>
                    </div>

                    {/* Comment body */}
                    <p className="text-sm text-gray-300 leading-relaxed">{c.comment}</p>
                  </motion.div>
                ))
              )}
            </div>

            {/* Footer hint */}
            <div className="px-4 py-3 border-t border-white/8">
              <p className="text-xs text-gray-600 text-center">
                Select any text in the chapter to comment on it
              </p>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
