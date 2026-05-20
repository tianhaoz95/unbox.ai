import {
  useState,
  useEffect,
  useRef,
  useCallback,
  type ReactNode,
} from "react";
import { motion, AnimatePresence } from "framer-motion";
import { MessageSquarePlus, X, Send, LogIn } from "lucide-react";
import { useAuth } from "../contexts/AuthContext";
import type { Comment } from "../hooks/useComments";

interface SelectionState {
  text: string;
  anchorText: string;
  x: number;
  y: number;
}

interface Props {
  children: ReactNode;
  comments: Comment[];
  onAddComment: (selectedText: string, anchorText: string, comment: string) => Promise<void>;
  onRequestSignIn: () => void;
}

// Build a surrounding-context string for disambiguation:
// grabs up to 30 chars before and after the selection within the same text node.
function buildAnchorText(selection: Selection): string {
  const range = selection.getRangeAt(0);
  const container = range.startContainer;
  const full = container.textContent ?? "";
  const start = Math.max(0, range.startOffset - 30);
  const end = Math.min(full.length, range.endOffset + 30);
  return full.slice(start, end);
}

// After comments load, scan text nodes inside `root` for each comment's
// selectedText and wrap the FIRST occurrence in a <mark> element.
// Skips nodes inside [data-no-highlight] containers (code blocks).
export function applyHighlights(root: HTMLElement, comments: Comment[]) {
  // Remove old marks first
  root.querySelectorAll("mark[data-comment-id]").forEach((el) => {
    const parent = el.parentNode;
    if (!parent) return;
    parent.replaceChild(document.createTextNode(el.textContent ?? ""), el);
    parent.normalize();
  });

  const seen = new Set<string>();

  for (const c of comments) {
    if (seen.has(c.selectedText)) continue; // only highlight first occurrence
    seen.add(c.selectedText);

    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        // Skip code blocks and interactive animations
        let el = node.parentElement;
        while (el && el !== root) {
          if (
            el.dataset.noHighlight ||
            el.tagName === "CODE" ||
            el.tagName === "PRE"
          )
            return NodeFilter.FILTER_REJECT;
          el = el.parentElement;
        }
        return NodeFilter.FILTER_ACCEPT;
      },
    });

    let node: Node | null;
    while ((node = walker.nextNode())) {
      const text = node.textContent ?? "";
      const idx = text.indexOf(c.selectedText);
      if (idx === -1) continue;

      // Split the text node and wrap the matched portion
      const before = text.slice(0, idx);
      const matched = text.slice(idx, idx + c.selectedText.length);
      const after = text.slice(idx + c.selectedText.length);

      const mark = document.createElement("mark");
      mark.dataset.commentId = c.id;
      mark.textContent = matched;
      mark.style.cssText =
        "background:rgba(14,165,233,0.25);border-bottom:2px solid rgba(14,165,233,0.7);border-radius:2px;cursor:pointer;";

      const parent = node.parentNode!;
      const frag = document.createDocumentFragment();
      if (before) frag.appendChild(document.createTextNode(before));
      frag.appendChild(mark);
      if (after) frag.appendChild(document.createTextNode(after));
      parent.replaceChild(frag, node);
      break; // only first occurrence per comment text
    }
  }
}

export function SelectionCommentLayer({
  children,
  comments,
  onAddComment,
  onRequestSignIn,
}: Props) {
  const { user } = useAuth();
  const containerRef = useRef<HTMLDivElement>(null);
  const [selection, setSelection] = useState<SelectionState | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [commentText, setCommentText] = useState("");
  const [submitting, setSubmitting] = useState(false);

  // Apply highlights whenever comments change
  useEffect(() => {
    if (!containerRef.current) return;
    applyHighlights(containerRef.current, comments);
  }, [comments]);

  const handleMouseUp = useCallback(() => {
    const sel = window.getSelection();
    if (!sel || sel.rangeCount === 0) return;
    const text = sel.toString().trim();
    if (text.length < 3) {
      setSelection(null);
      return;
    }

    // Don't trigger inside code blocks
    const range = sel.getRangeAt(0);
    let el = range.startContainer.parentElement;
    while (el) {
      if (el.tagName === "PRE" || el.tagName === "CODE" || el.dataset?.noHighlight) {
        setSelection(null);
        return;
      }
      el = el.parentElement;
    }

    const rect = range.getBoundingClientRect();
    const containerRect = containerRef.current?.getBoundingClientRect();
    if (!containerRect) return;

    setSelection({
      text,
      anchorText: buildAnchorText(sel),
      x: rect.left - containerRect.left + rect.width / 2,
      y: rect.top - containerRect.top - 8,
    });
    setShowForm(false);
  }, []);

  const handleSubmit = async () => {
    if (!selection || !commentText.trim()) return;
    if (!user) {
      onRequestSignIn();
      return;
    }
    setSubmitting(true);
    try {
      await onAddComment(selection.text, selection.anchorText, commentText.trim());
      setCommentText("");
      setSelection(null);
      setShowForm(false);
      window.getSelection()?.removeAllRanges();
    } finally {
      setSubmitting(false);
    }
  };

  const dismiss = () => {
    setSelection(null);
    setShowForm(false);
    setCommentText("");
    window.getSelection()?.removeAllRanges();
  };

  return (
    <div
      ref={containerRef}
      className="relative"
      onMouseUp={handleMouseUp}
    >
      {children}

      {/* Floating bubble on selection */}
      <AnimatePresence>
        {selection && !showForm && (
          <motion.div
            key="bubble"
            initial={{ opacity: 0, scale: 0.85, y: 4 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.85 }}
            transition={{ duration: 0.15 }}
            className="absolute z-50 flex items-center gap-1"
            style={{
              left: selection.x,
              top: selection.y,
              transform: "translate(-50%, -100%)",
            }}
          >
            <button
              onClick={() => setShowForm(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-brand-500 hover:bg-brand-600 text-white text-xs font-semibold shadow-lg transition-colors"
            >
              <MessageSquarePlus size={13} />
              Comment
            </button>
          </motion.div>
        )}

        {/* Comment form */}
        {selection && showForm && (
          <motion.div
            key="form"
            initial={{ opacity: 0, scale: 0.9, y: 4 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ duration: 0.18 }}
            className="absolute z-50 w-72 rounded-2xl bg-surface-800 border border-white/10 shadow-2xl p-4"
            style={{
              left: Math.min(
                Math.max(selection.x - 144, 0),
                (containerRef.current?.offsetWidth ?? 600) - 288
              ),
              top: selection.y,
              transform: "translateY(calc(-100% - 8px))",
            }}
          >
            {/* Selected text preview */}
            <div className="mb-3 px-2.5 py-1.5 rounded-lg bg-brand-500/10 border border-brand-500/20">
              <p className="text-xs text-brand-300 italic line-clamp-2">
                "{selection.text}"
              </p>
            </div>

            {user ? (
              <>
                <div className="flex items-center gap-2 mb-2">
                  <img
                    src={user.photoURL ?? ""}
                    alt=""
                    className="w-5 h-5 rounded-full"
                  />
                  <span className="text-xs text-gray-400">{user.displayName}</span>
                </div>
                <textarea
                  autoFocus
                  value={commentText}
                  onChange={(e) => setCommentText(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleSubmit();
                    if (e.key === "Escape") dismiss();
                  }}
                  placeholder="What's confusing or incorrect here?"
                  rows={3}
                  className="w-full bg-surface-700 border border-white/8 rounded-xl px-3 py-2 text-sm text-white placeholder-gray-600 resize-none focus:outline-none focus:border-brand-500/50 transition-colors"
                />
                <div className="flex items-center justify-between mt-2">
                  <span className="text-xs text-gray-600">⌘↵ to submit</span>
                  <div className="flex gap-2">
                    <button
                      onClick={dismiss}
                      className="p-1.5 rounded-lg text-gray-500 hover:text-white hover:bg-white/5 transition-colors"
                    >
                      <X size={14} />
                    </button>
                    <button
                      onClick={handleSubmit}
                      disabled={!commentText.trim() || submitting}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-brand-500 hover:bg-brand-600 disabled:opacity-40 disabled:cursor-not-allowed text-white text-xs font-semibold transition-colors"
                    >
                      <Send size={12} />
                      {submitting ? "Sending…" : "Send"}
                    </button>
                  </div>
                </div>
              </>
            ) : (
              <div className="text-center py-2">
                <p className="text-sm text-gray-400 mb-3">
                  Sign in to leave a comment
                </p>
                <button
                  onClick={() => {
                    dismiss();
                    onRequestSignIn();
                  }}
                  className="flex items-center gap-2 mx-auto px-4 py-2 rounded-xl bg-white text-gray-900 text-sm font-semibold hover:bg-gray-100 transition-colors"
                >
                  <LogIn size={14} />
                  Sign in with Google
                </button>
                <button
                  onClick={dismiss}
                  className="mt-2 text-xs text-gray-600 hover:text-gray-400"
                >
                  Cancel
                </button>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
