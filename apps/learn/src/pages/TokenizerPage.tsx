import { ChapterLayout } from "../components/ChapterLayout";
import { Section } from "../components/Section";
import { CodeBlock } from "../components/CodeBlock";
import { Callout } from "../components/Callout";
import { TokenizerAnim } from "../components/animations/TokenizerAnim";
import { BPEAnim } from "../components/animations/BPEAnim";

const bpeTrainCode = `# unbox_platform/tokenizer/train.py
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from pathlib import Path

def train_bpe_tokenizer(
    data_path: str,
    output_dir: str,
    vocab_size: int = 32_000,
) -> Tokenizer:
    """
    Train a Byte-Pair Encoding tokenizer on raw text.

    Why 32k vocab?
    - Too small (e.g. 1k): many tokens per word, slow inference, long sequences
    - Too large (e.g. 256k): large embedding table, rare tokens underfit
    - 32k–50k: the Goldilocks range used by LLaMA, Mistral, etc.
    """
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    # ByteLevel: operate on UTF-8 bytes, not characters.
    # This makes the tokenizer language-agnostic and handles any Unicode.
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,           # ignore pairs appearing < 2 times
        special_tokens=["<pad>", "<eos>", "<bos>", "<unk>"],
        show_progress=True,
    )

    # Feed an iterator of text strings — this streams the file
    def text_iterator():
        with open(data_path) as f:
            for line in f:
                doc = json.loads(line)
                yield doc["text"]

    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)

    # Add post-processing: automatically prepend <bos> and append <eos>
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[("<bos>", bos_id), ("<eos>", eos_id)],
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tokenizer.save(f"{output_dir}/tokenizer.json")
    print(f"Saved tokenizer to {output_dir}/  (vocab_size={tokenizer.get_vocab_size()})")
    return tokenizer
`;

const bpeAlgorithmCode = `# The BPE algorithm — conceptual implementation
def train_bpe(corpus: list[str], num_merges: int) -> dict[tuple, str]:
    """
    Byte Pair Encoding: start with individual characters,
    iteratively merge the most frequent adjacent pair.
    """
    # Step 1: split every word into characters + end marker
    vocab = {}
    for word in corpus:
        chars = list(word) + ["</w>"]
        vocab[" ".join(chars)] = corpus.count(word)

    merges = {}

    for _ in range(num_merges):
        # Step 2: count all adjacent pairs in the current vocabulary
        pairs = {}
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] = pairs.get(pair, 0) + freq

        if not pairs:
            break

        # Step 3: find the most frequent pair
        best_pair = max(pairs, key=pairs.get)
        merges[best_pair] = "".join(best_pair)  # record the merge

        # Step 4: apply the merge to the vocabulary
        new_vocab = {}
        bigram = " ".join(best_pair)
        merged = "".join(best_pair)
        for word in vocab:
            new_word = word.replace(bigram, merged)
            new_vocab[new_word] = vocab[word]
        vocab = new_vocab

    return merges  # these merges define the tokenizer
`;

const inferenceCode = `# unbox_platform/tokenizer/tokenizer.py
from tokenizers import Tokenizer as HFTokenizer

class UnboxTokenizer:
    """
    Thin wrapper around HuggingFace tokenizers.
    Handles encoding, decoding, and incremental streaming decode.
    """

    def __init__(self, tokenizer_dir: str):
        self._tok = HFTokenizer.from_file(f"{tokenizer_dir}/tokenizer.json")
        self.vocab_size = self._tok.get_vocab_size()
        self.eos_id = self._tok.token_to_id("<eos>")
        self.bos_id = self._tok.token_to_id("<bos>")
        self.pad_id = self._tok.token_to_id("<pad>")

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special)

    def decode_streaming(self, token_buffer: list[int]) -> tuple[str, list[int]]:
        """
        Incremental decode: return the longest confirmed text and the
        remaining buffered tokens.

        Multi-byte UTF-8 characters can span multiple tokens. We cannot
        decode a partial UTF-8 sequence — it would produce garbage.
        Strategy: try to decode the full buffer, fall back one token at a
        time until we get valid UTF-8.
        """
        for end in range(len(token_buffer), 0, -1):
            try:
                text = self._tok.decode(token_buffer[:end])
                text.encode("utf-8")   # validate it's complete UTF-8
                return text, token_buffer[end:]
            except (UnicodeDecodeError, Exception):
                continue
        return "", token_buffer   # nothing can be decoded yet
`;

const vocabSizeTable = [
  { model: "GPT-2", vocab: "50,257", notes: "BPE, character-level" },
  { model: "LLaMA 2", vocab: "32,000", notes: "SentencePiece BPE" },
  { model: "LLaMA 3 / Mistral", vocab: "128,256", notes: "Tiktoken-based" },
  { model: "unbox-760m", vocab: "32,000", notes: "HF BPE (this project)" },
  { model: "GPT-4 (est.)", vocab: "~100,000", notes: "cl100k_base tiktoken" },
];

export function TokenizerPage() {
  return (
    <ChapterLayout
      num="02"
      title="Tokenizer"
      subtitle="Text is a string. Models need integers. The tokenizer is the bridge — and its design cascades through every downstream decision."
      color="text-emerald-400"
      prev={{ path: "/data", label: "Data Pipeline" }}
      next={{ path: "/pretraining", label: "Pre-training" }}
    >
      {/* Live demo */}
      <div className="grid sm:grid-cols-2 gap-4">
        <TokenizerAnim />
        <BPEAnim />
      </div>

      <Section stepNum={1} title="What is a tokenizer?">
        <p className="prose-custom text-base">
          A tokenizer splits text into <strong>tokens</strong> — discrete units that
          map to integers from a fixed vocabulary. The model never sees characters or
          bytes directly; it operates entirely on token IDs.
        </p>

        <p className="prose-custom text-base">
          The vocabulary is learned from training data. Common subwords get their own
          ID; rare words are split into multiple tokens. "tokenization" might become
          <code>["token", "ization"]</code>. "ChatGPT" might be{" "}
          <code>["Chat", "G", "PT"]</code>.
        </p>

        <div className="card-glass p-5">
          <div className="text-sm font-semibold text-white mb-3">
            Industry vocabulary sizes
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/5">
                  <th className="text-left py-2 text-gray-500 font-medium">Model</th>
                  <th className="text-left py-2 text-gray-500 font-medium">Vocab size</th>
                  <th className="text-left py-2 text-gray-500 font-medium hidden sm:table-cell">Notes</th>
                </tr>
              </thead>
              <tbody>
                {vocabSizeTable.map((row) => (
                  <tr key={row.model} className="border-b border-white/3">
                    <td className="py-2 text-gray-300 font-mono text-xs">{row.model}</td>
                    <td className="py-2 text-brand-300 font-mono text-xs">{row.vocab}</td>
                    <td className="py-2 text-gray-500 text-xs hidden sm:table-cell">{row.notes}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <Callout type="why">
          <strong>Why not just split on words or characters?</strong> Word-level
          vocabularies explode in size (millions of words, OOV problem). Character-level
          vocabularies are tiny but sequences become very long — 4× longer than BPE,
          meaning 4× more attention operations. BPE finds the sweet spot.
        </Callout>
      </Section>

      <Section stepNum={2} title="The BPE algorithm">
        <p className="prose-custom text-base">
          Byte Pair Encoding (BPE) builds a vocabulary by starting with individual
          characters and iteratively merging the <strong>most frequent adjacent pair</strong>.
          After N merges, you have a vocabulary of N + (initial alphabet size) tokens.
        </p>

        <CodeBlock
          code={bpeAlgorithmCode}
          filename="BPE conceptual implementation"
          highlights={[17, 18, 19, 20, 23, 24, 26, 27, 28]}
        />

        <Callout type="insight">
          The merge order matters and is deterministic given the corpus. When you
          save a tokenizer, you're saving the ordered list of merges. At inference
          time, you apply those same merges greedily to encode new text. This is why
          tokenizers are <strong>corpus-specific</strong> — a tokenizer trained on
          English code will be inefficient for Japanese text.
        </Callout>

        <div className="grid sm:grid-cols-2 gap-4">
          {[
            {
              title: "BPE strengths",
              color: "#10b981",
              items: [
                "Language-agnostic at byte level",
                "Handles any Unicode gracefully",
                "Vocabulary is interpretable",
                "Fast encoding (O(n log n))",
              ],
            },
            {
              title: "SentencePiece differences",
              color: "#0ea5e9",
              items: [
                "Works on raw text (no pre-tokenization)",
                "Handles spaces as explicit tokens",
                "Unigram Language Model variant",
                "Used by LLaMA, T5, ALBERT",
              ],
            },
          ].map((card) => (
            <div
              key={card.title}
              className="card-glass p-4"
              style={{ borderColor: `${card.color}20` }}
            >
              <h4
                className="font-semibold text-sm mb-3"
                style={{ color: card.color }}
              >
                {card.title}
              </h4>
              <ul className="space-y-1.5">
                {card.items.map((item) => (
                  <li
                    key={item}
                    className="flex items-center gap-2 text-sm text-gray-400"
                  >
                    <span style={{ color: card.color }}>→</span>
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </Section>

      <Section stepNum={3} title="Train your tokenizer">
        <p className="prose-custom text-base">
          We use the HuggingFace <code>tokenizers</code> library — it's written in Rust
          and trains a 32k vocab tokenizer in under 2 minutes on 10GB of text.
        </p>

        <CodeBlock
          code={bpeTrainCode}
          filename="unbox_platform/tokenizer/train.py"
          highlights={[16, 17, 18, 23, 24, 25, 26, 31]}
        />

        <Callout type="warning">
          Train the tokenizer <strong>before</strong> tokenizing your data (Chapter 01).
          The tokenizer is a prerequisite for the data pipeline. In practice, you train
          the tokenizer on a representative sample (e.g. 1B tokens), then tokenize the
          full dataset with the resulting vocabulary.
        </Callout>

        <CodeBlock
          language="bash"
          code={`# Train the tokenizer (run first, before data pipeline)
.venv/bin/python -m unbox_platform.tokenizer.train \\
    --data data/fineweb_edu_10bt.jsonl \\
    --output checkpoints/tokenizer \\
    --vocab_size 32000

# Output:
# Training: 100%|████████| 1.2M/1.2M [01:43<00:00]
# Saved tokenizer to checkpoints/tokenizer/  (vocab_size=32000)

# Inspect the vocabulary
python -c "
from tokenizers import Tokenizer
t = Tokenizer.from_file('checkpoints/tokenizer/tokenizer.json')
print(t.get_vocab_size())           # 32000
print(t.encode('Hello world').ids)  # [15496, 995]
print(t.decode([15496, 995]))       # 'Hello world'
"
`}
          filename="terminal"
        />
      </Section>

      <Section stepNum={4} title="Using the tokenizer at inference">
        <p className="prose-custom text-base">
          During inference, the tokenizer has an extra responsibility: incremental
          streaming decode. The model generates one token at a time, but we can't always
          decode a single token to a character — some characters span multiple tokens in
          multi-byte UTF-8 encoding.
        </p>

        <CodeBlock
          code={inferenceCode}
          filename="unbox_platform/tokenizer/tokenizer.py"
          highlights={[36, 37, 38, 39, 40, 41, 42, 43]}
        />

        <Callout type="insight">
          This is why streaming LLM outputs sometimes have a short delay before the
          first characters appear — the server is buffering tokens until it can confirm
          a complete UTF-8 sequence. The same issue occurs with special characters in
          Chinese, Arabic, emoji, etc.
        </Callout>
      </Section>

      <Section stepNum={5} title="Key design decisions">
        <div className="grid sm:grid-cols-2 gap-4">
          {[
            {
              question: "Vocab size: 32k vs 128k?",
              answer:
                "32k is the LLaMA 2 sweet spot. 128k (LLaMA 3) trades a larger embedding table for shorter sequences in multilingual settings. For English-only research, 32k is more parameter-efficient.",
              color: "#0ea5e9",
            },
            {
              question: "Add domain tokens?",
              answer:
                'Yes if you have domain-specific strings that fragment badly — e.g., "<|endoftext|>", code keywords, or math notation. Add them as special tokens before training.',
              color: "#a855f7",
            },
            {
              question: "Case sensitivity?",
              answer:
                "Keep it. Lowercasing loses information the model can learn from. BPE naturally handles case variants as separate tokens ('The' vs 'the'), and the model learns they're related.",
              color: "#10b981",
            },
            {
              question: "Re-use a pretrained tokenizer?",
              answer:
                "Absolutely fine for research. Using LLaMA's tokenizer saves you the training step and ensures compatibility with pretrained weights for fine-tuning experiments.",
              color: "#f59e0b",
            },
          ].map((item) => (
            <div key={item.question} className="card-glass p-4">
              <div
                className="text-sm font-semibold mb-2"
                style={{ color: item.color }}
              >
                {item.question}
              </div>
              <p className="text-sm text-gray-400 leading-relaxed">{item.answer}</p>
            </div>
          ))}
        </div>
      </Section>
    </ChapterLayout>
  );
}
