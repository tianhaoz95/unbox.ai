import { Routes, Route } from "react-router-dom";
import { Navbar } from "./components/Navbar";
import { HomePage } from "./pages/HomePage";
import { DataPage } from "./pages/DataPage";
import { TokenizerPage } from "./pages/TokenizerPage";
import { PretrainingPage } from "./pages/PretrainingPage";
import { ComingSoonPage } from "./pages/ComingSoonPage";

export default function App() {
  return (
    <div className="min-h-screen bg-surface-900">
      <Navbar />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/data" element={<DataPage />} />
        <Route path="/tokenizer" element={<TokenizerPage />} />
        <Route path="/pretraining" element={<PretrainingPage />} />
        <Route path="/sft" element={<ComingSoonPage chapter="SFT" />} />
        <Route path="/rl" element={<ComingSoonPage chapter="RL Post-training" />} />
        <Route path="/inference" element={<ComingSoonPage chapter="Inference" />} />
        <Route path="/distillation" element={<ComingSoonPage chapter="Distillation" />} />
      </Routes>
    </div>
  );
}
