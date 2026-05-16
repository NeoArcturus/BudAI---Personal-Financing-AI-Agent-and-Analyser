"use client";

import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Button,
  Link,
  Accordion,
  Chip,
} from "@heroui/react";
import {
  Zap,
  TrendingUp,
  Sparkles,
  Shield,
  Lock,
  EyeOff,
  Activity,
  Database,
  BrainCircuit,
  Menu,
  ChevronDown,
} from "lucide-react";

export default function LandingPage() {
  const router = useRouter();
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    const token = localStorage.getItem("budai_token");
    if (token) {
      router.push("/home");
    } else {

      const timer = setTimeout(() => setIsLoaded(true), 0);
      return () => clearTimeout(timer);
    }
  }, [router]);

  if (!isLoaded) return null;

  return (
    <div className="min-h-screen bg-[#08090D] text-[#dce4e5] selection:bg-cyan-500/30 selection:text-cyan-200 bg-grid-pattern relative overflow-x-hidden">

      <div className="fixed inset-0 z-0 pointer-events-none opacity-50 mix-blend-screen">
        <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] rounded-full bg-cyan-500/20 blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[60%] h-[60%] rounded-full bg-pink-500/10 blur-[150px]" />
      </div>

      <nav className="fixed top-0 w-full z-50 h-20 obsidian-glass border-b border-white/5 flex justify-between items-center px-6 md:px-10">
        <div className="flex items-center gap-2">
          <Activity className="text-cyan-400 w-8 h-8" />
          <span className="font-bold text-white text-xl tracking-tight">
            BudAI
          </span>
        </div>

        <div className="hidden md:flex items-center gap-10">
          <Link
            className="text-sm font-medium text-white/60 hover:text-cyan-400 transition-colors"
            href="#features"
          >
            Features
          </Link>
          <Link
            className="text-sm font-medium text-white/60 hover:text-cyan-400 transition-colors"
            href="#engine"
          >
            Intelligence Engine
          </Link>
          <Link
            className="text-sm font-medium text-white/60 hover:text-cyan-400 transition-colors"
            href="#security"
          >
            Security
          </Link>
          <Link
            className="text-sm font-medium text-white/60 hover:text-cyan-400 transition-colors"
            href="#faq"
          >
            FAQ
          </Link>
        </div>

        <div className="flex items-center gap-4">
          <Link
            href="/login"
            className="hidden md:block text-sm font-medium text-white/60 hover:text-white px-4 transition-colors"
          >
            Login
          </Link>
          <Button
            onPress={() => router.push("/register")}
            className="bg-cyan-400 text-black font-bold px-6 h-10 rounded-lg neon-glow-primary hover:bg-cyan-300"
          >
            Get Started
          </Button>
          <Button
            isIconOnly
            variant="ghost"
            className="md:hidden text-white/60"
          >
            <Menu size={24} />
          </Button>
        </div>
      </nav>

      <main className="relative z-10 pt-20">

        <section className="relative min-h-[80vh] flex items-center justify-center px-6 py-24 bg-hero-abstract">
          <div className="max-w-4xl mx-auto text-center flex flex-col items-center gap-8">
            <div className="inline-flex items-center gap-2 px-4 py-1 rounded-full border border-cyan-400/30 bg-cyan-400/10 text-cyan-400 text-xs font-bold tracking-wider">
              <Zap size={14} className="fill-current" />
              <span>V2.0 ORCHESTRATION ENGINE LIVE</span>
            </div>

            <h1 className="text-4xl md:text-6xl font-bold tracking-tight text-white leading-tight">
              Your Financial Intelligence, <br />
              <span className="text-transparent bg-clip-text bg-linear-to-r from-cyan-400 to-cyan-200">
                Evolutionized
              </span>
            </h1>

            <p className="text-lg md:text-xl text-white/60 max-w-2xl mx-auto leading-relaxed">
              BudAI integrates directly with your financial ecosystem to provide
              actionable insights, predictive forecasting, and empathetic
              advisory through advanced multi-agent orchestration.
            </p>

            <div className="flex flex-col sm:flex-row items-center gap-4 mt-4 w-full sm:w-auto">
              <Button
                onPress={() => router.push("/register")}
                className="bg-cyan-400 text-black font-bold px-10 h-14 text-base rounded-xl neon-glow-primary hover:bg-cyan-300 w-full sm:w-auto"
              >
                Get Started Free
              </Button>
              <Button
                variant="outline"
                className="border-white/10 text-cyan-400 px-10 h-14 text-base rounded-xl obsidian-glass hover:bg-white/5 w-full sm:w-auto"
              >
                Book a Demo
              </Button>
            </div>
          </div>

          <div className="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-1/2 w-full max-w-6xl h-64 obsidian-glass rounded-t-[3rem] border-t border-x border-white/10 opacity-60 flex justify-center pt-8 overflow-hidden pointer-events-none">
            <div className="w-3/4 h-full border border-white/5 rounded-t-2xl bg-white/5 flex p-6 gap-6">
              <div className="w-1/4 h-full bg-white/5 rounded-xl animate-pulse" />
              <div className="w-3/4 h-full bg-white/5 rounded-xl flex flex-col gap-4 p-4 relative">
                <div className="absolute top-4 right-4 bg-cyan-400/20 border border-cyan-400/30 text-cyan-400 px-3 py-1 rounded-full text-xs font-bold flex items-center gap-2">
                  <TrendingUp size={14} /> Wealth Velocity +12%
                </div>
                <div className="w-full h-8 bg-white/5 rounded-lg" />
                <div className="w-full flex-1 bg-white/5 rounded-lg" />
              </div>
            </div>
          </div>
        </section>

        <section className="py-20 border-y border-white/5 bg-black/20 overflow-hidden relative z-20">
          <div className="text-center mb-10">
            <h3 className="text-xs font-bold text-white/40 uppercase tracking-[0.3em]">
              Universal Connectivity
            </h3>
          </div>
          <div className="relative flex overflow-x-hidden w-full">
            <div className="flex animate-marquee whitespace-nowrap gap-20 items-center py-4 opacity-40 px-10">
              {[
                "HSBC",
                "BARCLAYS",
                "MONZO",
                "REVOLUT",
                "LLOYDS",
                "CHASE",
                "STARLING",
                "HSBC",
                "BARCLAYS",
                "MONZO",
                "REVOLUT",
                "LLOYDS",
                "CHASE",
                "STARLING",
              ].map((bank, i) => (
                <span
                  key={i}
                  className="text-3xl font-black text-white tracking-tighter italic"
                >
                  {bank}
                </span>
              ))}
            </div>
          </div>
        </section>

        <section id="engine" className="py-32 px-6 relative z-20">
          <div className="max-w-6xl mx-auto flex flex-col items-center">
            <div className="text-center mb-20">
              <h2 className="text-3xl md:text-5xl font-bold text-white mb-4">
                The Intelligence Engine
              </h2>
              <p className="text-lg text-white/60 max-w-2xl mx-auto">
                Experience the liquid state dashboard. Real-time data mapped to
                actionable strategies.
              </p>
            </div>

            <div className="w-full aspect-video max-w-5xl obsidian-glass rounded-3xl border border-white/10 p-4 relative overflow-hidden group shadow-2xl">
              <div className="w-full h-full bg-[#0d1516]/40 rounded-2xl flex border border-white/5">

                <div className="w-1/4 border-r border-white/5 p-6 hidden md:flex flex-col gap-6">
                  <div className="h-8 bg-white/10 rounded-lg w-1/2 mb-4" />
                  <div className="h-4 bg-white/5 rounded-lg w-3/4" />
                  <div className="h-4 bg-white/5 rounded-lg w-full" />
                  <div className="h-4 bg-white/5 rounded-lg w-5/6" />
                </div>

                <div className="flex-1 p-8 flex flex-col gap-8 relative">

                  <div className="absolute top-1/4 left-1/4 bg-cyan-400/20 border border-cyan-400/30 text-cyan-400 px-4 py-2 rounded-full text-sm font-bold flex items-center gap-2 shadow-lg z-10 hover:scale-105 transition-transform cursor-default">
                    <TrendingUp size={18} /> Wealth Velocity +12.4%
                  </div>
                  <div className="absolute bottom-1/3 right-1/4 bg-pink-500/20 border border-pink-500/30 text-pink-500 px-4 py-2 rounded-full text-sm font-bold flex items-center gap-2 shadow-lg z-10 hover:scale-105 transition-transform cursor-default">
                    <Activity size={18} /> Unusual Subscription: $49.99
                  </div>

                  <div className="grid grid-cols-3 gap-6">
                    <div className="h-32 bg-white/5 rounded-2xl border border-white/5 animate-pulse" />
                    <div className="h-32 bg-white/5 rounded-2xl border border-white/5" />
                    <div className="h-32 bg-white/5 rounded-2xl border border-white/5" />
                  </div>

                  <div className="flex-1 bg-white/5 rounded-2xl border border-white/5 p-6">
                    <div className="w-full h-full border-b border-l border-white/10 flex items-end p-4 gap-4">
                      {[25, 50, 35, 75, 90, 60, 45].map((h, i) => (
                        <div
                          key={i}
                          className="flex-1 bg-cyan-400/20 rounded-t-lg border-t border-x border-cyan-400/30 transition-all duration-500 hover:bg-cyan-400/40"
                          style={{ height: `${h}%` }}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section id="features" className="py-32 px-6 relative z-20 bg-black/20">
          <div className="max-w-6xl mx-auto">
            <div className="text-center mb-20">
              <h2 className="text-3xl md:text-5xl font-bold text-white mb-4">
                How it Works
              </h2>
              <p className="text-lg text-white/60">
                Three phases to total financial clarity.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-10 relative">
              <div className="hidden md:block absolute top-1/2 left-0 w-full h-px bg-linear-to-r from-transparent via-cyan-400/20 to-transparent -translate-y-1/2 z-0" />

              {[
                {
                  step: "01",
                  title: "Connect",
                  icon: Database,
                  desc: "Securely link your accounts using read-only API access backed by AES-256 encryption.",
                },
                {
                  step: "02",
                  title: "Analyze",
                  icon: BrainCircuit,
                  desc: "Multi-agent clustering algorithms instantly categorize transactions and build your unique financial graph.",
                },
                {
                  step: "03",
                  title: "Evolve",
                  icon: Sparkles,
                  desc: "Receive proactive, predictive strategies to optimize cash flow and accelerate wealth creation.",
                },
              ].map((item, i) => (
                <div
                  key={i}
                  className="obsidian-glass rounded-3xl p-10 relative z-10 card-hover flex flex-col gap-6"
                >
                  <div className="text-cyan-400 text-6xl font-black opacity-10 absolute top-4 right-6">
                    {item.step}
                  </div>
                  <div className="w-14 h-14 rounded-2xl bg-cyan-400/10 border border-cyan-400/20 flex items-center justify-center text-cyan-400">
                    <item.icon size={28} />
                  </div>
                  <h3 className="text-2xl font-bold text-white">
                    {item.title}
                  </h3>
                  <p className="text-white/60 leading-relaxed">{item.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section id="security" className="py-32 px-6 relative z-20">
          <div className="max-w-6xl mx-auto obsidian-glass rounded-[3rem] p-12 md:p-20 border border-white/10 flex flex-col md:flex-row gap-16 items-center">
            <div className="flex-1 flex flex-col gap-8 text-center md:text-left items-center md:items-start">
              <Chip
                className="border-cyan-400/30 bg-cyan-400/10 text-cyan-400 rounded-lg flex items-center gap-2"
                variant="secondary"
              >
                <Shield size={14} />
                <span>ENTERPRISE GRADE</span>
              </Chip>
              <h2 className="text-3xl md:text-5xl font-bold text-white">
                Zero-Compromise Security
              </h2>
              <p className="text-lg text-white/60 max-w-xl">
                We believe your financial data belongs to you. We employ
                bank-level encryption, enforce strict MFA, and operate on a
                strict no-sell data policy.
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-8 w-full">
                <div className="flex items-start gap-4">
                  <div className="p-2 rounded-lg bg-cyan-400/10 text-cyan-400">
                    <Lock size={20} />
                  </div>
                  <div>
                    <h4 className="font-bold text-white">AES-256 Encryption</h4>
                    <p className="text-sm text-white/40">
                      Military-grade protection for data at rest and in transit.
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <div className="p-2 rounded-lg bg-cyan-400/10 text-cyan-400">
                    <EyeOff size={20} />
                  </div>
                  <div>
                    <h4 className="font-bold text-white">No-Sell Policy</h4>
                    <p className="text-sm text-white/40">
                      Your data is never monetized or shared with third parties.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="w-full md:w-1/3 flex justify-center">
              <div className="relative w-64 h-64 flex items-center justify-center">
                <svg className="absolute inset-0 w-full h-full -rotate-90">
                  <circle
                    cx="128"
                    cy="128"
                    r="120"
                    fill="none"
                    stroke="rgba(0,229,255,0.1)"
                    strokeWidth="12"
                  />
                  <circle
                    cx="128"
                    cy="128"
                    r="120"
                    fill="none"
                    stroke="#00e5ff"
                    strokeWidth="12"
                    strokeDasharray="754"
                    strokeDashoffset="0"
                    className="drop-shadow-[0_0_15px_rgba(0,229,255,0.6)]"
                  />
                </svg>
                <div className="text-center z-10">
                  <span className="block text-6xl font-black text-white">
                    100
                  </span>
                  <span className="text-xs text-cyan-400 font-bold tracking-[0.2em] uppercase">
                    Security Score
                  </span>
                </div>
              </div>
            </div>
          </div>
        </section>

        <div className="w-full bg-black/40 py-4 border-y border-white/5 overflow-hidden flex z-30 relative">
          <div className="flex animate-marquee whitespace-nowrap gap-16 items-center text-xs font-mono text-white/40 uppercase tracking-widest">
            {[
              "Transactions Processed: 4,291,003",
              "Inference Latency: 42ms",
              "Active Agents: 12,045",
              "Uptime: 99.99%",
              "Financial Graph Nodes: 1.2B",
            ].map((stat, i) => (
              <span key={i} className="flex items-center gap-3">
                <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse shadow-[0_0_8px_rgba(0,229,255,0.8)]" />
                {stat}
              </span>
            ))}

            {[
              "Transactions Processed: 4,291,003",
              "Inference Latency: 42ms",
              "Active Agents: 12,045",
              "Uptime: 99.99%",
              "Financial Graph Nodes: 1.2B",
            ].map((stat, i) => (
              <span key={`dup-${i}`} className="flex items-center gap-3 ml-16">
                <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse shadow-[0_0_8px_rgba(0,229,255,0.8)]" />
                {stat}
              </span>
            ))}
          </div>
        </div>

        <section id="faq" className="py-32 px-6 relative z-20">
          <div className="max-w-3xl mx-auto">
            <div className="text-center mb-16">
              <h2 className="text-3xl md:text-5xl font-bold text-white">
                Common Queries
              </h2>
            </div>

            <div className="flex flex-col gap-4">
              {[
                {
                  id: "1",
                  title: "How secure is my financial data?",
                  content:
                    "We use AES-256 encryption, enforce strict MFA, and have a zero-sell data policy. We only maintain read-only access to your accounts via official banking APIs.",
                },
                {
                  id: "2",
                  title: "Which banks do you support?",
                  content:
                    "We support over 10,000 financial institutions globally, including major UK/EU banks like HSBC, Monzo, Revolut, and Barclays via Open Banking protocols.",
                },
                {
                  id: "3",
                  title: "How accurate is the categorization?",
                  content:
                    "Our multi-agent orchestration engine achieves 99.9% categorization accuracy by analyzing merchant data, location context, and historical spending patterns.",
                },
              ].map((faq) => (
                <Accordion
                  key={faq.id}
                  className="obsidian-glass border border-white/10 rounded-2xl overflow-hidden"
                >
                  <Accordion.Item>
                    <Accordion.Heading>
                      <Accordion.Trigger className="w-full p-6 font-bold text-white flex justify-between items-center hover:bg-white/5 transition-colors">
                        {faq.title}
                        <ChevronDown size={20} className="text-cyan-400" />
                      </Accordion.Trigger>
                    </Accordion.Heading>
                    <Accordion.Panel className="px-6 pb-6 text-white/60">
                      <div className="pt-4 border-t border-white/5">
                        {faq.content}
                      </div>
                    </Accordion.Panel>
                  </Accordion.Item>
                </Accordion>
              ))}
            </div>
          </div>
        </section>
      </main>

      <footer className="border-t border-white/5 bg-[#08090D] pt-24 pb-12 px-6 relative z-20">
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-16 mb-20">
            <div className="lg:col-span-2 flex flex-col gap-8">
              <div className="flex items-center gap-3">
                <Activity className="text-cyan-400 w-8 h-8" />
                <span className="text-2xl font-bold text-white tracking-tight">
                  BudAI
                </span>
              </div>
              <p className="text-white/40 max-w-xs leading-relaxed">
                Advanced financial intelligence for the modern professional.
                Evolutionize your wealth tracking with multi-agent
                orchestration.
              </p>
              <div className="flex items-center gap-3 text-xs font-bold text-green-500 bg-green-500/10 border border-green-500/20 px-4 py-2 rounded-full w-fit">
                <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.6)]" />
                SYSTEM STATUS: ALL SYSTEMS OPERATIONAL
              </div>
            </div>

            <div className="flex flex-col gap-6">
              <h4 className="text-white font-bold text-sm uppercase tracking-widest">
                Product
              </h4>
              <Link
                href="#"
                className="text-white/40 hover:text-cyan-400 transition-colors text-sm"
              >
                Engine
              </Link>
              <Link
                href="#"
                className="text-white/40 hover:text-cyan-400 transition-colors text-sm"
              >
                Forecasting
              </Link>
              <Link
                href="#"
                className="text-white/40 hover:text-cyan-400 transition-colors text-sm"
              >
                Security
              </Link>
              <Link
                href="#"
                className="text-white/40 hover:text-cyan-400 transition-colors text-sm"
              >
                Pricing
              </Link>
            </div>

            <div className="flex flex-col gap-6">
              <h4 className="text-white font-bold text-sm uppercase tracking-widest">
                Resources
              </h4>
              <Link
                href="#"
                className="text-white/40 hover:text-cyan-400 transition-colors text-sm"
              >
                Docs
              </Link>
              <Link
                href="#"
                className="text-white/40 hover:text-cyan-400 transition-colors text-sm"
              >
                API
              </Link>
              <Link
                href="#"
                className="text-white/40 hover:text-cyan-400 transition-colors text-sm"
              >
                Community
              </Link>
              <Link
                href="#"
                className="text-white/40 hover:text-cyan-400 transition-colors text-sm"
              >
                Blog
              </Link>
            </div>

            <div className="flex flex-col gap-6">
              <h4 className="text-white font-bold text-sm uppercase tracking-widest">
                Legal
              </h4>
              <Link
                href="#"
                className="text-white/40 hover:text-cyan-400 transition-colors text-sm"
              >
                Privacy
              </Link>
              <Link
                href="#"
                className="text-white/40 hover:text-cyan-400 transition-colors text-sm"
              >
                Terms
              </Link>
              <Link
                href="#"
                className="text-white/40 hover:text-cyan-400 transition-colors text-sm"
              >
                Cookies
              </Link>
            </div>
          </div>

          <div className="flex flex-col md:flex-row justify-between items-center pt-8 border-t border-white/5 gap-6 text-white/30 text-xs font-medium">
            <p>© 2024 BudAI Inc. Evolutionizing Intelligence.</p>
            <div className="flex gap-8">
              <Link
                href="#"
                className="text-inherit hover:text-white transition-colors"
              >
                TWITTER
              </Link>
              <Link
                href="#"
                className="text-inherit hover:text-white transition-colors"
              >
                LINKEDIN
              </Link>
              <Link
                href="#"
                className="text-inherit hover:text-white transition-colors"
              >
                GITHUB
              </Link>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
