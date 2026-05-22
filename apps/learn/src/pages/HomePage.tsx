import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowRight, CheckCircle, Clock, Zap, BookOpen, Code, Brain } from "lucide-react";
import { useLanguage } from "../contexts/LanguageContext";

const chapterMeta = [
  { num: "01", path: "/data",        ready: true,  color: "#6366f1", icon: "🌐", key: "ch01" },
  { num: "02", path: "/tokenizer",   ready: true,  color: "#10b981", icon: "✂️",  key: "ch02" },
  { num: "03", path: "/pretraining", ready: true,  color: "#0ea5e9", icon: "⚡",  key: "ch03" },
  { num: "04", path: "/eval",        ready: true,  color: "#8b5cf6", icon: "📊", key: "ch04" },
  { num: "05", path: "/sft",         ready: true,  color: "#f59e0b", icon: "💬", key: "ch05" },
  { num: "06", path: "/dpo",         ready: true,  color: "#ec4899", icon: "🏆", key: "ch06" },
  { num: "07", path: "/inference",   ready: false, color: "#a855f7", icon: "🚀", key: "ch07" },
  { num: "08", path: "/distillation",ready: false, color: "#14b8a6", icon: "🔬", key: "ch08" },
];

const statsKeys = ["lines", "anims", "chapters", "scratch"] as const;
const statsValues = ["10,000+", "20+", "7", "100%"];

export function HomePage() {
  const { t } = useLanguage();

  const chapters = chapterMeta.map((ch) => ({
    ...ch,
    title: t(`${ch.key}.title`),
    desc: t(`${ch.key}.desc`),
    topics: [t(`${ch.key}.t1`), t(`${ch.key}.t2`), t(`${ch.key}.t3`), t(`${ch.key}.t4`)],
  }));

  const stats = statsKeys.map((k, i) => ({
    label: t(`home.stats.${k}`),
    value: statsValues[i],
  }));

  const philosophy = [
    { title: t("home.simplicity.title"), desc: t("home.simplicity.desc") },
    { title: t("home.realCode.title"),   desc: t("home.realCode.desc") },
    { title: t("home.build.title"),      desc: t("home.build.desc") },
  ];

  return (
    <div>
      {/* Hero */}
      <section className="relative overflow-hidden">
        {/* Background glow */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-20 left-1/4 w-96 h-96 bg-brand-500/8 rounded-full blur-3xl" />
          <div className="absolute top-40 right-1/4 w-64 h-64 bg-cyan-500/6 rounded-full blur-3xl" />
        </div>

        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 pb-16 relative">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center"
          >
            {/* Badge */}
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.1 }}
              className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-brand-500/10 border border-brand-500/25 text-brand-300 text-xs font-semibold mb-6"
            >
              <Zap size={12} />
              {t("home.badge")}
            </motion.div>

            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold text-white mb-6 leading-tight tracking-tight">
              {t("home.title")}{" "}
              <span className="gradient-text">{t("home.gradient")}</span>
            </h1>

            <p className="text-xl text-gray-400 max-w-2xl mx-auto mb-8 leading-relaxed">
              {t("home.subtitle")}
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                to="/data"
                className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-brand-500 hover:bg-brand-600 text-white font-semibold text-sm transition-colors glow-brand-sm"
              >
                {t("home.start")}
                <ArrowRight size={16} />
              </Link>
              <a
                href="https://github.com/tianhaoz95/unbox.ai"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-white/5 hover:bg-white/10 text-white font-semibold text-sm transition-colors border border-white/10"
              >
                <Code size={16} />
                {t("home.source")}
              </a>
            </div>
          </motion.div>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="mt-16 grid grid-cols-2 sm:grid-cols-4 gap-4"
          >
            {stats.map((s) => (
              <div key={s.label} className="card-glass p-4 text-center">
                <div className="text-2xl font-bold gradient-text mb-1">{s.value}</div>
                <div className="text-xs text-gray-500">{s.label}</div>
              </div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Learning path */}
      <section className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="flex items-center gap-3 mb-8">
          <BookOpen size={18} className="text-brand-400" />
          <h2 className="text-2xl font-bold text-white">{t("home.path")}</h2>
        </div>

        <div className="space-y-4">
          {chapters.map((ch, i) => (
            <motion.div
              key={ch.num}
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.08 }}
            >
              <Link
                to={ch.path}
                className={`group block card-glass p-5 hover:bg-surface-700/60 transition-all duration-200 gradient-border ${
                  !ch.ready ? "opacity-60 pointer-events-none" : ""
                }`}
              >
                <div className="flex items-start gap-4">
                  {/* Chapter icon */}
                  <div
                    className="w-12 h-12 rounded-xl flex items-center justify-center text-2xl flex-shrink-0"
                    style={{ background: `${ch.color}15`, border: `1px solid ${ch.color}30` }}
                  >
                    {ch.icon}
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-3 mb-1.5">
                      <span className="font-mono text-xs text-gray-600">{ch.num}</span>
                      <h3 className="font-bold text-white text-lg group-hover:text-brand-300 transition-colors">
                        {ch.title}
                      </h3>
                      {ch.ready ? (
                        <span className="flex items-center gap-1 text-xs text-emerald-400">
                          <CheckCircle size={12} />
                          {t("home.ready")}
                        </span>
                      ) : (
                        <span className="flex items-center gap-1 text-xs text-gray-600">
                          <Clock size={12} />
                          {t("home.soon")}
                        </span>
                      )}
                    </div>

                    <p className="text-sm text-gray-400 mb-3">{ch.desc}</p>

                    <div className="flex flex-wrap gap-2">
                      {ch.topics.map((topic) => (
                        <span
                          key={topic}
                          className="px-2 py-0.5 rounded-full text-xs border"
                          style={{
                            color: `${ch.color}cc`,
                            borderColor: `${ch.color}25`,
                            background: `${ch.color}0a`,
                          }}
                        >
                          {topic}
                        </span>
                      ))}
                    </div>
                  </div>

                  {ch.ready && (
                    <ArrowRight
                      size={18}
                      className="text-gray-600 group-hover:text-brand-400 group-hover:translate-x-1 transition-all flex-shrink-0 mt-1"
                    />
                  )}
                </div>
              </Link>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Philosophy */}
      <section className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12 pb-20">
        <div className="card-glass p-8 gradient-border">
          <div className="flex items-center gap-2 mb-4">
            <Brain size={18} className="text-brand-400" />
            <h2 className="text-xl font-bold text-white">{t("home.philosophy.title")}</h2>
          </div>
          <div className="grid sm:grid-cols-3 gap-6">
            {philosophy.map((item) => (
              <div key={item.title}>
                <h3 className="font-semibold text-white mb-2">{item.title}</h3>
                <p className="text-sm text-gray-400 leading-relaxed">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
