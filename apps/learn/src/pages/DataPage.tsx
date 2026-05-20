import { ChapterLayout } from "../components/ChapterLayout";
import { Section } from "../components/Section";
import { CodeBlock } from "../components/CodeBlock";
import { Callout } from "../components/Callout";
import { DataPipelineAnim } from "../components/animations/DataPipelineAnim";

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
  return (
    <ChapterLayout
      num="01"
      title="Data Pipeline"
      subtitle="From raw web crawl to packed token sequences ready for training. This is the unsexy foundation everything else depends on."
      color="text-indigo-400"
      next={{ path: "/tokenizer", label: "Tokenizer" }}
    >
      {/* Overview animation */}
      <DataPipelineAnim />

      <Section stepNum={1} title="Where does the data come from?">
        <p className="prose-custom text-base">
          Pre-training data is the single biggest lever on model quality. You need{" "}
          <strong>trillions of tokens</strong> of diverse, high-quality text. The industry
          standard is a filtered subset of{" "}
          <strong>Common Crawl</strong> — a petabyte-scale snapshot of the web taken
          monthly since 2008.
        </p>
        <p className="prose-custom text-base">
          For this project we use <strong>FineWeb-Edu</strong>: a 1.3 trillion token
          dataset filtered to educational content using a classifier trained on human
          ratings. It's freely available on Hugging Face and produces much better
          models per token than raw crawl data.
        </p>

        <Callout type="why">
          <strong>Why not just use Wikipedia or books?</strong> They're clean but tiny
          — Wikipedia is ~4B tokens, Project Gutenberg ~3B. A 760M parameter model needs
          ~15B+ tokens to converge (Chinchilla scaling). You need web data.
        </Callout>

        <CodeBlock
          code={dataPrepareCode}
          filename="unbox_platform/data/prepare.py"
          highlights={[6, 7, 8, 9, 14, 15]}
        />

        <Callout type="tip">
          Use <code>streaming=True</code> when downloading large datasets. It lets you
          start processing immediately without downloading the full ~200GB to disk first.
        </Callout>
      </Section>

      <Section stepNum={2} title="Data quality: filter and deduplicate">
        <p className="prose-custom text-base">
          Raw web data is noisy. Pages contain spam, boilerplate, duplicated content, and
          text in unexpected languages. Two operations have the highest ROI:
        </p>

        <div className="grid sm:grid-cols-2 gap-4">
          {[
            {
              title: "Quality filtering",
              color: "#f59e0b",
              items: [
                "Language detection (fasttext)",
                "Perplexity filtering with KenLM",
                "Remove HTML artifacts",
                "Filter short / boilerplate text",
              ],
            },
            {
              title: "Deduplication",
              color: "#ec4899",
              items: [
                "Exact dedup with MD5/SHA-256",
                "Near-dedup with MinHash LSH",
                "URL-level dedup",
                "Paragraph-level exact match",
              ],
            },
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
          Deduplication is disproportionately impactful. GPT-3's training data was ~3%
          duplicates, but removing them improved validation loss more than adding
          equivalent unique tokens. The model memorizes duplicates instead of learning.
        </Callout>
      </Section>

      <Section stepNum={3} title="Tokenize and pack sequences">
        <p className="prose-custom text-base">
          After filtering, we convert raw text to token IDs and pack them into
          fixed-length sequences. This is a one-time offline step that produces the
          exact numpy array we load during training.
        </p>

        <Callout type="why">
          <strong>Why pack instead of pad?</strong> Padding wastes computation on tokens
          that contribute zero gradient. With padding you might hit 60-70% token
          utilization. Sequence packing with document boundaries gives you{" "}
          <strong>~100% utilization</strong>. At scale this is the difference between
          a $1M training run and a $600K one.
        </Callout>

        <CodeBlock
          code={tokenizerDataCode}
          filename="unbox_platform/data/pipeline.py"
          highlights={[14, 15, 16, 17, 28, 29, 30]}
        />

        <div className="card-glass p-5">
          <div className="text-sm font-semibold text-white mb-3">Sequence packing visualized</div>
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
              <span className="text-xs text-gray-500 w-20 flex-shrink-0">Packed</span>
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
          <p className="text-xs text-gray-500 mt-2">
            All 3 documents packed into one 1024-token sequence with <code>&lt;eos&gt;</code> separators.
            100% token utilization.
          </p>
        </div>
      </Section>

      <Section stepNum={4} title="Build the DataLoader">
        <p className="prose-custom text-base">
          The final step is loading the packed numpy arrays into a PyTorch{" "}
          <code>DataLoader</code>. We use <strong>memory mapping</strong> so the OS
          pages in only the chunks needed — no full dataset in RAM.
        </p>

        <CodeBlock
          code={dataLoaderCode}
          filename="unbox_platform/data/loader.py"
          highlights={[13, 14, 23, 24, 25]}
        />

        <Callout type="tip">
          <strong>num_workers=4</strong> launches 4 background processes that prefetch
          batches while the GPU is busy on the previous step. Without this, CPU data loading
          becomes the bottleneck. Set <code>pin_memory=True</code> to enable faster
          CPU→GPU DMA transfers.
        </Callout>

        <div className="grid sm:grid-cols-3 gap-4">
          {[
            { key: "Batch size", val: "32–512", note: "per GPU, tune to memory" },
            { key: "Seq length", val: "1024–8192", note: "longer = more compute" },
            { key: "Token utilization", val: "~100%", note: "vs 60-70% with padding" },
          ].map((item) => (
            <div key={item.key} className="card-glass p-4">
              <div className="text-xs text-gray-500 mb-1">{item.key}</div>
              <div className="text-xl font-bold text-brand-300 font-mono">{item.val}</div>
              <div className="text-xs text-gray-600 mt-1">{item.note}</div>
            </div>
          ))}
        </div>
      </Section>

      <Section stepNum={5} title="Run it">
        <p className="prose-custom text-base">
          The full pipeline is wired up as a CLI command. Run it once before training:
        </p>

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
          You only run the data pipeline <strong>once</strong>. The resulting{" "}
          <code>.npy</code> file is your training data for all experiments. If you change
          the tokenizer vocabulary, you need to re-tokenize. If you just change model
          architecture or training hyperparameters, you don't.
        </Callout>
      </Section>
    </ChapterLayout>
  );
}
