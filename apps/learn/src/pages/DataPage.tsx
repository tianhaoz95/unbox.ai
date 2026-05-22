import { ChapterLayout } from "../components/ChapterLayout";
import { Section } from "../components/Section";
import { CodeBlock } from "../components/CodeBlock";
import { Callout } from "../components/Callout";
import { DataPipelineAnim } from "../components/animations/DataPipelineAnim";
import { useLanguage } from "../contexts/LanguageContext";

const dataPrepareCode = `# unbox_platform/data/prepare.py
import json
from datasets import load_dataset
from pathlib import Path

def download_fineweb_edu(output_path: str, sample: str = "sample-10BT") -> None:
    """
    Download FineWeb-Edu — a high-quality 1.3T token subset of CommonCrawl
    filtered for educational content using a classifier.
    """
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=sample,       # "sample-10BT" = 10 billion tokens, good for experiments
        split="train",
        streaming=True,    # stream to avoid loading 1.3T tokens into RAM
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        for doc in ds:
            # Each doc has: text, url, score, token_count, ...
            f.write(json.dumps({"text": doc["text"]}) + "\\n")
`;

const deduplicationCode = `# unbox_platform/data/dedup.py
import hashlib
from typing import Iterator

def minhash_dedup(
    docs: Iterator[dict],
    threshold: float = 0.8,
    num_perm: int = 128,
) -> Iterator[dict]:
    """
    MinHash-based near-deduplication.
    Two documents with Jaccard similarity > threshold are considered duplicates.

    Why MinHash? Exact dedup (MD5) misses paraphrases and near-copies.
    MinHash approximates Jaccard similarity in O(n) time using sketches.
    """
    from datasketch import MinHash, MinHashLSH

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    seen = 0

    for i, doc in enumerate(docs):
        text = doc["text"]
        m = MinHash(num_perm=num_perm)

        # Shingling: slide a 3-word window across the text
        words = text.lower().split()
        for shingle in zip(words, words[1:], words[2:]):
            m.update(" ".join(shingle).encode("utf8"))

        key = f"doc_{i}"
        if not lsh.query(m):   # no duplicate found
            lsh.insert(key, m)
            yield doc
        else:
            seen += 1

    print(f"Removed {seen} near-duplicate documents")
`;

const tokenizerDataCode = `# unbox_platform/data/pipeline.py
from pathlib import Path
from tokenizers import Tokenizer

def tokenize_dataset(
    input_path: str,
    output_path: str,
    tokenizer_path: str,
    seq_len: int = 1024,
) -> None:
    """
    Tokenize documents and pack them into fixed-length sequences.

    Key design decision: NO padding. Instead we concatenate documents
    with an <eos> separator and slice into fixed-length chunks.
    This achieves ~100% token utilization vs ~60-70% with padding.
    """
    tokenizer = Tokenizer.from_file(tokenizer_path)
    eos_id = tokenizer.token_to_id("<eos>")

    buffer: list[int] = []
    sequences: list[list[int]] = []

    with open(input_path) as f:
        for line in f:
            doc = json.loads(line)
            ids = tokenizer.encode(doc["text"]).ids
            buffer.extend(ids)
            buffer.append(eos_id)   # document boundary marker

            # Slice off complete chunks as soon as we have enough
            while len(buffer) >= seq_len + 1:   # +1 for the shifted label
                sequences.append(buffer[:seq_len + 1])
                buffer = buffer[seq_len + 1:]

    # Save as numpy for fast memory-mapped loading
    import numpy as np
    arr = np.array(sequences, dtype=np.uint16)   # uint16 fits vocab ≤ 65536
    np.save(output_path, arr)
    print(f"Saved {len(sequences):,} sequences to {output_path}")
`;

const dataLoaderCode = `# unbox_platform/data/loader.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PackedTokenDataset(Dataset):
    """
    Memory-mapped dataset over pre-tokenized packed sequences.
    Memory mapping avoids loading the entire dataset into RAM —
    the OS pages in only the chunks actually needed.
    """

    def __init__(self, path: str):
        # np.load with mmap_mode doesn't load data into RAM
        self.data = np.load(path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = torch.from_numpy(self.data[idx].astype(np.int64))
        return {
            "input_ids": seq[:-1],   # tokens 0..T-2 → input
            "labels": seq[1:],       # tokens 1..T-1 → target (teacher forcing)
        }


def build_dataloader(path: str, batch_size: int, num_workers: int = 4) -> DataLoader:
    dataset = PackedTokenDataset(path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,    # speed up CPU→GPU transfer
    )
`;

export function DataPage() {
  const { t } = useLanguage();

  const filterItems = [t("data.s2.filter.i1"), t("data.s2.filter.i2"), t("data.s2.filter.i3"), t("data.s2.filter.i4")];
  const dedupItems  = [t("data.s2.dedup.i1"),  t("data.s2.dedup.i2"),  t("data.s2.dedup.i3"),  t("data.s2.dedup.i4")];

  return (
    <ChapterLayout
      num="01"
      title={t("ch01.title")}
      subtitle={t("data.subtitle")}
      color="text-indigo-400"
      next={{ path: "/tokenizer", label: t("ch02.title") }}
    >
      {/* Overview animation */}
      <DataPipelineAnim />

      <Section stepNum={1} title={t("data.s1.title")}>
        <p className="prose-custom text-base" dangerouslySetInnerHTML={{ __html: t("data.s1.p1") }} />
        <p className="prose-custom text-base" dangerouslySetInnerHTML={{ __html: t("data.s1.p2") }} />

        <Callout type="why">
          <span dangerouslySetInnerHTML={{ __html: t("data.s1.why") }} />
        </Callout>

        <CodeBlock
          code={dataPrepareCode}
          filename="unbox_platform/data/prepare.py"
          highlights={[6, 7, 8, 9, 14, 15]}
        />

        <Callout type="tip">
          <span dangerouslySetInnerHTML={{ __html: t("data.s1.tip") }} />
        </Callout>
      </Section>

      <Section stepNum={2} title={t("data.s2.title")}>
        <p className="prose-custom text-base" dangerouslySetInnerHTML={{ __html: t("data.s2.p1") }} />

        <div className="grid sm:grid-cols-2 gap-4">
          {[
            { title: t("data.s2.filter.title"), color: "#f59e0b", items: filterItems },
            { title: t("data.s2.dedup.title"),  color: "#ec4899", items: dedupItems  },
          ].map((card) => (
            <div
              key={card.title}
              className="card-glass p-4"
              style={{ borderColor: `${card.color}20` }}
            >
              <h4 className="font-semibold text-sm mb-3" style={{ color: card.color }}>
                {card.title}
              </h4>
              <ul className="space-y-1.5">
                {card.items.map((item) => (
                  <li key={item} className="flex items-center gap-2 text-sm text-gray-400">
                    <span style={{ color: card.color }}>→</span>
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <CodeBlock
          code={deduplicationCode}
          filename="unbox_platform/data/dedup.py"
          highlights={[13, 14, 15, 16, 17]}
        />

        <Callout type="insight">
          {t("data.s2.insight")}
        </Callout>
      </Section>

      <Section stepNum={3} title={t("data.s3.title")}>
        <p className="prose-custom text-base" dangerouslySetInnerHTML={{ __html: t("data.s3.p1") }} />

        <Callout type="why">
          <span dangerouslySetInnerHTML={{ __html: t("data.s3.why") }} />
        </Callout>

        <CodeBlock
          code={tokenizerDataCode}
          filename="unbox_platform/data/pipeline.py"
          highlights={[14, 15, 16, 17, 28, 29, 30]}
        />

        <div className="card-glass p-5">
          <div className="text-sm font-semibold text-white mb-3">{t("data.s3.vis.title")}</div>
          <div className="space-y-2">
            {[
              { label: "Document A", len: 60, color: "#0ea5e9" },
              { label: "Document B", len: 25, color: "#10b981" },
              { label: "Document C", len: 15, color: "#a855f7" },
            ].map((doc) => (
              <div key={doc.label} className="flex items-center gap-3">
                <span className="text-xs text-gray-500 w-20 flex-shrink-0">{doc.label}</span>
                <div className="flex-1 h-6 bg-surface-700 rounded overflow-hidden relative">
                  <div
                    className="h-full rounded"
                    style={{
                      width: `${doc.len}%`,
                      background: `${doc.color}40`,
                      borderRight: `2px solid ${doc.color}`,
                    }}
                  />
                  <span
                    className="absolute inset-y-0 left-2 flex items-center text-xs font-mono"
                    style={{ color: doc.color }}
                  >
                    {doc.len}% of 1024
                  </span>
                </div>
              </div>
            ))}
            <div className="flex items-center gap-3">
              <span className="text-xs text-gray-500 w-20 flex-shrink-0">{t("data.s3.vis.packed")}</span>
              <div className="flex-1 h-6 bg-surface-700 rounded overflow-hidden">
                {[
                  { len: 60, color: "#0ea5e9" },
                  { len: 25, color: "#10b981" },
                  { len: 15, color: "#a855f7" },
                ].map((seg, i) => (
                  <div
                    key={i}
                    className="h-full float-left"
                    style={{
                      width: `${seg.len}%`,
                      background: `${seg.color}40`,
                      borderRight: `1px solid ${seg.color}60`,
                    }}
                  />
                ))}
              </div>
            </div>
          </div>
          <p className="text-xs text-gray-500 mt-2" dangerouslySetInnerHTML={{ __html: t("data.s3.vis.note") }} />
        </div>
      </Section>

      <Section stepNum={4} title={t("data.s4.title")}>
        <p className="prose-custom text-base" dangerouslySetInnerHTML={{ __html: t("data.s4.p1") }} />

        <CodeBlock
          code={dataLoaderCode}
          filename="unbox_platform/data/loader.py"
          highlights={[13, 14, 23, 24, 25]}
        />

        <Callout type="tip">
          <span dangerouslySetInnerHTML={{ __html: t("data.s4.tip") }} />
        </Callout>

        <div className="grid sm:grid-cols-3 gap-4">
          {([
            { k: t("data.s4.stat1.k"), v: t("data.s4.stat1.v"), n: t("data.s4.stat1.n") },
            { k: t("data.s4.stat2.k"), v: t("data.s4.stat2.v"), n: t("data.s4.stat2.n") },
            { k: t("data.s4.stat3.k"), v: t("data.s4.stat3.v"), n: t("data.s4.stat3.n") },
          ]).map((item) => (
            <div key={item.k} className="card-glass p-4">
              <div className="text-xs text-gray-500 mb-1">{item.k}</div>
              <div className="text-xl font-bold text-brand-300 font-mono">{item.v}</div>
              <div className="text-xs text-gray-600 mt-1">{item.n}</div>
            </div>
          ))}
        </div>
      </Section>

      <Section stepNum={5} title={t("data.s5.title")}>
        <p className="prose-custom text-base" dangerouslySetInnerHTML={{ __html: t("data.s5.p1") }} />

        <CodeBlock
          language="bash"
          code={`# Download FineWeb-Edu sample-10BT (~200GB)
.venv/bin/python -m unbox_platform.data.prepare \\
    --output data/fineweb_edu_10bt.jsonl

# Tokenize and pack (uses pre-trained tokenizer from Chapter 02)
.venv/bin/python -m unbox_platform.data.pipeline \\
    --input data/fineweb_edu_10bt.jsonl \\
    --output data/packed_1024.npy \\
    --tokenizer checkpoints/tokenizer \\
    --seq_len 1024

# Verify
python -c "import numpy as np; d=np.load('data/packed_1024.npy'); print(f'{len(d):,} sequences, {len(d)*1024/1e9:.1f}B tokens')"
`}
          filename="terminal"
        />

        <Callout type="insight">
          <span dangerouslySetInnerHTML={{ __html: t("data.s5.insight") }} />
        </Callout>
      </Section>
    </ChapterLayout>
  );
}
