import { useState } from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { Copy, Check } from "lucide-react";

interface CodeBlockProps {
  code: string;
  language?: string;
  filename?: string;
  highlights?: number[];
}

export function CodeBlock({ code, language = "python", filename, highlights = [] }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const customStyle = {
    ...oneDark,
    'pre[class*="language-"]': {
      ...oneDark['pre[class*="language-"]'],
      background: "transparent",
      margin: 0,
      padding: "1.25rem",
      fontSize: "0.8rem",
      lineHeight: "1.7",
    },
    'code[class*="language-"]': {
      ...oneDark['code[class*="language-"]'],
      background: "transparent",
      fontSize: "0.8rem",
    },
  };

  return (
    <div className="rounded-xl overflow-hidden border border-white/8 bg-surface-800">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-white/8 bg-surface-700/50">
        <div className="flex items-center gap-2">
          <div className="flex gap-1.5">
            <div className="w-3 h-3 rounded-full bg-red-500/70" />
            <div className="w-3 h-3 rounded-full bg-yellow-500/70" />
            <div className="w-3 h-3 rounded-full bg-green-500/70" />
          </div>
          {filename && (
            <span className="font-mono text-xs text-gray-500 ml-2">{filename}</span>
          )}
        </div>
        <button
          onClick={copy}
          className="flex items-center gap-1.5 px-2 py-1 rounded-md text-xs text-gray-500 hover:text-gray-300 hover:bg-white/5 transition-colors"
        >
          {copied ? (
            <>
              <Check size={12} className="text-green-400" />
              <span className="text-green-400">Copied</span>
            </>
          ) : (
            <>
              <Copy size={12} />
              Copy
            </>
          )}
        </button>
      </div>

      {/* Code */}
      <div className="overflow-x-auto">
        <SyntaxHighlighter
          language={language}
          style={customStyle}
          wrapLines={true}
          lineProps={(lineNumber) => ({
            style: highlights.includes(lineNumber)
              ? { backgroundColor: "rgba(14, 165, 233, 0.12)", display: "block", margin: "0 -1.25rem", padding: "0 1.25rem" }
              : {},
          })}
          customStyle={{ background: "transparent", padding: 0 }}
        >
          {code.trim()}
        </SyntaxHighlighter>
      </div>
    </div>
  );
}
