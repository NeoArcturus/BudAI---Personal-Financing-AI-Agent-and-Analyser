"use client";

import React from "react";
import { useRouter } from "next/navigation";
import { Button, Link, Accordion, Tabs } from "@heroui/react";
import {
  TrendingUp,
  Sparkles,
  Shield,
  Lock,
  EyeOff,
  Activity,
  Database,
  Menu,
  ChevronDown,
  MessageSquare,
  ArrowRight,
  Search,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import Image from "next/image";

export default function LandingPage() {
  const tabs = [
    { id: "overview", label: "Overview" },
    { id: "credit", label: "Credit Health" },
    { id: "budget", label: "Budgets" },
    { id: "stresstest", label: "Scenarios" },
    { id: "benchmarks", label: "Benchmarks" },
    { id: "goals", label: "Goals" },
    { id: "tco", label: "Asset Costs" },
  ];
  const [activeTab, setActiveTab] = React.useState("overview");
  const [isAutoPlaying, setIsAutoPlaying] = React.useState(true);

  React.useEffect(() => {
    if (!isAutoPlaying) return;
    const interval = setInterval(() => {
      setActiveTab((current) => {
        const idx = tabs.findIndex((t) => t.id === current);
        return tabs[(idx + 1) % tabs.length].id;
      });
    }, 6000);
    return () => clearInterval(interval);
  }, [isAutoPlaying]);
  const router = useRouter();

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.15,
        delayChildren: 0.2,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.8,
        ease: [0.16, 1, 0.3, 1] as const,
      },
    },
  };

  return (
    <div className="dark min-h-screen text-foreground selection:bg-primary/30 selection:text-primary relative overflow-x-hidden bg-transparent font-sans">
      <nav className="fixed top-0 w-full h-24 bg-[#0c131d] border-b-[0.5px] border-white/5 flex justify-between items-center px-6 md:px-10 transition-all">
        <div className="flex items-center gap-2 shrink-0">
          <Image
            src="/FullLogo.jpg"
            alt="BudAI Logo"
            width={80}
            height={25}
            className="rounded-sm object-contain"
            priority
          />
        </div>

        <div className="hidden md:flex items-center gap-10">
          <Link
            className="text-[10px] font-bold uppercase tracking-[0.2em] text-foreground/40 hover:text-primary transition-all"
            href="#features"
          >
            Capabilities
          </Link>
          <Link
            className="text-[10px] font-bold uppercase tracking-[0.2em] text-foreground/40 hover:text-primary transition-all"
            href="#dashboard"
          >
            Dashboard
          </Link>
          <Link
            className="text-[10px] font-bold uppercase tracking-[0.2em] text-foreground/40 hover:text-primary transition-all"
            href="#security"
          >
            Security
          </Link>
          <Link
            className="text-[10px] font-bold uppercase tracking-[0.2em] text-foreground/40 hover:text-primary transition-all"
            href="#faq"
          >
            Support
          </Link>
        </div>

        <div className="flex items-center gap-4">
          <Link
            href="/login"
            className="hidden md:block text-[10px] font-bold uppercase tracking-widest text-foreground/40 hover:text-foreground px-4 transition-all"
          >
            Log In
          </Link>
          <Button
            onPress={() => router.push("/onboarding")}
            variant="secondary" className="font-extrabold text-xs tracking-wide px-6 h-9 rounded-md transition-all "
          >
            Get Started
          </Button>
          <Button
            isIconOnly
            variant="ghost"
            className="md:hidden text-foreground/60 border-none"
          >
            <Menu size={20} />
          </Button>
        </div>
      </nav>

      <main className="relative z-10 pt-24">
        <section className="relative min-h-[90vh] flex items-center justify-center px-6 py-24">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="max-w-4xl mx-auto text-center flex flex-col items-center gap-10"
          >
            <motion.h1
              variants={itemVariants}
              className="text-5xl md:text-7xl font-mono font-black tracking-tighter text-foreground leading-tight uppercase border-l-4 border-primary pl-6 text-left w-full"
            >
              Personal Financial<br />
              <span className="text-primary tracking-widest drop-shadow-[0_0_15px_rgba(var(--nextui-primary),0.3)]">
                Analysis
              </span>
            </motion.h1>

            <motion.p
              variants={itemVariants}
              className="text-[11px] md:text-xs text-foreground/50 max-w-xl font-mono uppercase tracking-[0.2em] leading-relaxed text-left w-full"
            >
              Institutional-grade financial analysis and transaction categorization.
            </motion.p>

            <motion.div
              variants={itemVariants}
              className="flex flex-col sm:flex-row items-center gap-6 mt-4 w-full justify-start"
            >
              <Button
                onPress={() => router.push("/onboarding")}
                variant="secondary" className="font-mono font-black tracking-[0.2em] text-[10px] uppercase px-12 h-14 rounded-xl "
              >
                Get Started
              </Button>
              <Button
                variant="outline"
                className="border-white/10 text-foreground/60 px-12 h-14 text-[10px] font-mono font-bold uppercase tracking-[0.2em] rounded-xl bg-white/5 backdrop-blur-xl hover:bg-white/10 hover:text-foreground transition-all"
              >
                Learn More
              </Button>
            </motion.div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 0.6, y: 0 }}
            transition={{ duration: 1.2, delay: 0.8, ease: [0.16, 1, 0.3, 1] }}
            className="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-1/2 w-full max-w-5xl h-64 bg-black/40 backdrop-blur-3xl rounded-t-xl border-t-[0.5px] border-x-[0.5px] border-white/10 flex justify-center pt-10 overflow-hidden pointer-events-none"
          >
            <motion.div
              animate={{ y: [0, -10, 0] }}
              transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
              className="w-4/5 h-full border-[0.5px] border-white/10 rounded-t-xl bg-primary/[0.02] flex p-8 gap-8 shadow-inner"
            >
              <div className="w-1/3 h-full bg-white/5 rounded-xl" />
              <div className="w-2/3 h-full bg-white/5 rounded-xl flex flex-col gap-6 p-6 relative">
                <div className="absolute top-6 right-6 bg-primary/10 border border-primary/20 text-primary px-4 py-1.5 rounded-lg text-[9px] font-black uppercase tracking-[0.2em] flex items-center gap-2">
                  <TrendingUp size={12} /> Target: +85%
                </div>
                <div className="w-1/2 h-6 bg-white/5 rounded-lg" />
                <div className="w-full flex-1 bg-white/5 rounded-lg" />
              </div>
            </motion.div>
          </motion.div>
        </section>


        <section className="py-24 bg-background border-y border-white/5 relative z-20">
          <div className="text-center mb-12">
            <h3 className="font-mono text-[10px] font-black text-white/40 tracking-[0.2em] uppercase">
              Bank Integrations
            </h3>
          </div>
          <div className="flex flex-wrap justify-center gap-12 md:gap-24 items-center opacity-40 px-6 max-w-6xl mx-auto">
            {["HSBC", "BARCLAYS", "MONZO", "REVOLUT", "LLOYDS", "CHASE"].map(
              (bank, i) => (
                <span
                  key={i}
                  className="text-2xl font-bold text-foreground tracking-tight"
                >
                  {bank}
                </span>
              ),
            )}
          </div>
        </section>

        <section id="dashboard" className="py-40 px-6 relative z-20">
          <div className="max-w-6xl mx-auto flex flex-col items-center">
            <div className="text-center mb-16">
              <motion.h2
                initial={{ opacity: 0, scale: 0.95 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                className="text-4xl md:text-5xl font-extrabold text-white mb-6 tracking-tight leading-tight drop-shadow-md"
              >
                Platform Interface
              </motion.h2>
              <p className="text-lg text-foreground/60 max-w-2xl mx-auto font-medium tracking-wide">
                Explore the different views and tools available in the application.
              </p>
            </div>


            <div className="w-full" onMouseEnter={() => setIsAutoPlaying(false)} onMouseLeave={() => setIsAutoPlaying(true)}>
              <div className="w-full overflow-x-auto scrollbar-hide border-b border-white/10 pb-4 mb-8">
                <div
                  className="w-max mx-auto flex justify-center gap-8 md:gap-12"
                  role="tablist"
                >
                  {tabs.map((tab: { id: string; label: string }) => (
                    <button
                      key={tab.id}
                      role="tab"
                      aria-selected={activeTab === tab.id}
                      onClick={() => {
                        setActiveTab(tab.id);
                        setIsAutoPlaying(false);
                      }}
                      className={`relative px-2 py-1 text-[11px] font-mono uppercase tracking-[0.2em] font-black transition-colors ${activeTab === tab.id ? "text-[#00f2ff]" : "text-white/40 hover:text-white/70"
                        }`}
                    >
                      {tab.label}
                      {activeTab === tab.id && (
                        <motion.div
                          layoutId="activeTabIndicator"
                          className="absolute left-0 right-0 h-0.5 bg-[#00f2ff] bottom-[-20px]"
                          transition={{ type: "spring", stiffness: 300, damping: 30 }}
                        />
                      )}
                      <div className="absolute inset-0 bg-[#00f2ff]/20 blur-xl rounded-full opacity-0 hover:opacity-100 transition-opacity pointer-events-none" />
                    </button>
                  ))}
                </div>
              </div>
              <AnimatePresence mode="wait">


                {/* Tab 1: Financial Overview */}
                {activeTab === "overview" && (
                  <motion.div
                    key="overview"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3 }}
                    className="outline-none w-full"
                  >
                    <div className="text-center mb-8">
                      <p className="font-mono text-xs text-white/50 uppercase tracking-widest">View your account balances and monthly spending.</p>
                    </div>
                    <div className="w-full aspect-video max-w-5xl mx-auto bg-black/40 backdrop-blur-3xl rounded-[3rem] border border-white/10 p-8 relative overflow-hidden shadow-2xl flex flex-col gap-6 text-left">
                      <div className="flex gap-6 h-2/5">
                        <div className="flex-1 bg-white/3 border-[0.5px] border-primary/30 rounded-2xl p-6 flex flex-col justify-between relative overflow-hidden shadow-inner group">
                          <div className="absolute inset-0 bg-primary/10 transition-opacity" />
                          <div className="relative z-10 flex justify-between items-start">
                            <div className="flex flex-col gap-1">
                              <span className="text-[9px] font-black text-foreground/40 uppercase tracking-[0.3em]">Live Account Balance</span>
                              <h2 className="text-4xl font-normal tracking-tighter text-foreground mt-1 font-mono">£14,250.50</h2>
                            </div>
                            <div className="w-12 h-12 bg-foreground text-background rounded-xl flex items-center justify-center font-black text-xl shadow-xl">B</div>
                          </div>
                          <div className="relative z-10 flex justify-between items-end mt-auto">
                            <span className="text-lg font-bold tracking-tight uppercase">BARCLAYS</span>
                            <div className="flex items-center gap-4 text-foreground/40 text-[10px] font-bold tracking-[0.2em] font-mono">
                              <span>****1234</span><span className="opacity-20">|</span><span>20-45-14</span>
                            </div>
                          </div>
                        </div>
                        <div className="w-1/3 bg-white/5 border-[0.5px] border-white/10 rounded-2xl p-6 flex flex-col justify-between">
                          <span className="text-[9px] font-black text-foreground/40 uppercase tracking-[0.3em]">Monthly Spend</span>
                          <h2 className="text-2xl font-semibold tracking-tight text-foreground font-mono">£2,450.00</h2>
                          <div className="flex gap-2 mt-4 items-end h-16 w-full border-b border-l border-white/10 p-2">
                            {[30, 50, 40, 80, 60, 90, 45].map((h, i) => (
                              <div key={i} className="flex-1 bg-primary/40 rounded-t-sm border-t border-x border-primary/20" style={{ height: `${h}%` }} />
                            ))}
                          </div>
                        </div>
                      </div>
                      <div className="flex-1 bg-white/5 border-[0.5px] border-white/10 rounded-2xl p-6 flex flex-col gap-4">
                        <span className="text-[9px] font-black text-foreground/40 uppercase tracking-[0.3em]">Recent Transactions</span>
                        <div className="flex flex-col gap-3 flex-1 overflow-hidden">
                          {[
                            { name: "Tesco Extra", category: "Groceries", amount: "-£45.20", date: "Today, 14:30" },
                            { name: "Transport for London", category: "Travel", amount: "-£12.50", date: "Today, 08:45" },
                          ].map((tx, i) => (
                            <div key={i} className="flex justify-between items-center bg-white/5 p-4 rounded-xl border border-white/5">
                              <div className="flex gap-4 items-center">
                                <div className="w-10 h-10 rounded-lg bg-white/10 flex items-center justify-center">
                                  <Activity size={16} className="text-foreground/40" />
                                </div>
                                <div className="flex flex-col">
                                  <span className="text-sm font-semibold tracking-tight">{tx.name}</span>
                                  <div className="flex items-center gap-2">
                                    <span className="text-[9px] text-foreground/40 uppercase tracking-widest">{tx.category}</span>
                                    <span className="text-[9px] text-foreground/20">•</span>
                                    <span className="text-[9px] text-foreground/40">{tx.date}</span>
                                  </div>
                                </div>
                              </div>
                              <span className="text-sm font-mono font-semibold tracking-tight">{tx.amount}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}


                {/* Tab 3: Credit Health */}
                {activeTab === "credit" && (
                  <motion.div
                    key="credit"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3 }}
                    className="outline-none w-full"
                  >
                    <div className="text-center mb-8">
                      <p className="font-mono text-xs text-white/50 uppercase tracking-widest">Track credit usage and debt.</p>
                    </div>
                    <div className="w-full aspect-video max-w-5xl mx-auto bg-black/40 backdrop-blur-3xl rounded-[3rem] border border-white/10 p-8 relative overflow-hidden shadow-2xl flex flex-col gap-6 text-left">
                      <div className="flex gap-6 h-1/2">
                        <div className="flex-[2] bg-white/5 border-[0.5px] border-white/10 rounded-2xl p-6 flex flex-col justify-between">
                          <span className="text-[9px] font-black text-foreground/40 uppercase tracking-[0.3em]">Total Credit Utilization</span>
                          <div className="flex flex-col gap-2 mt-4">
                            <div className="flex justify-between items-end">
                              <span className="text-4xl font-mono text-foreground tracking-tighter">24.5%</span>
                              <span className="text-[10px] font-mono text-[#00f2ff] uppercase tracking-widest">Optimal &lt; 30%</span>
                            </div>
                            <div className="w-full h-2 bg-white/5 rounded-full overflow-hidden mt-2">
                              <div className="h-full bg-[#00f2ff] rounded-full" style={{ width: '24.5%' }} />
                            </div>
                            <div className="flex justify-between mt-1 text-[9px] font-mono text-white/40">
                              <span>£3,675 Used</span><span>£15,000 Limit</span>
                            </div>
                          </div>
                        </div>
                        <div className="flex-1 bg-white/5 border-[0.5px] border-white/10 rounded-2xl p-6 flex flex-col justify-center gap-2">
                          <span className="text-[9px] font-black text-foreground/40 uppercase tracking-[0.3em]">Debt-to-Income</span>
                          <span className="text-4xl font-mono text-[#7000ff] tracking-tighter">18%</span>
                          <span className="text-[9px] font-mono text-white/40">of Monthly Income</span>
                        </div>
                      </div>
                      <div className="flex-1 bg-white/5 border-[0.5px] border-white/10 rounded-2xl p-6 flex flex-col gap-4">
                        <div className="flex justify-between items-center border-b border-white/10 pb-4">
                          <span className="text-[9px] font-black text-foreground/40 uppercase tracking-[0.3em]">Active Credit Accounts</span>
                        </div>
                        <div className="flex flex-col gap-3">
                          {[
                            { name: "Amex Platinum", apr: "24.9%", balance: "£1,250.00", limit: "£8,000.00" },
                            { name: "Barclaycard", apr: "19.9%", balance: "£2,425.00", limit: "£7,000.00" },
                          ].map((acc, i) => (
                            <div key={i} className="flex justify-between items-center bg-black/40 p-4 rounded-xl border border-white/5">
                              <div className="flex gap-4 items-center">
                                <div className="w-8 h-8 rounded-md bg-white/10 flex items-center justify-center text-white/40 border border-white/5"><Lock size={14} /></div>
                                <div className="flex flex-col">
                                  <span className="text-sm font-semibold tracking-tight">{acc.name}</span>
                                  <span className="text-[10px] font-mono text-white/40 uppercase tracking-widest">APR: {acc.apr}</span>
                                </div>
                              </div>
                              <div className="flex items-center gap-6">
                                <div className="flex flex-col items-end">
                                  <span className="text-[9px] font-black text-foreground/40 uppercase tracking-[0.3em]">Balance</span>
                                  <span className="font-mono text-lg">{acc.balance}</span>
                                </div>
                                <div className="flex flex-col items-end hidden md:flex">
                                  <span className="text-[9px] font-black text-foreground/40 uppercase tracking-[0.3em]">Limit</span>
                                  <span className="font-mono text-white/40">{acc.limit}</span>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}

                {/* Tab 4: Budget Variance */}
                {activeTab === "budget" && (
                  <motion.div
                    key="budget"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3 }}
                    className="outline-none w-full"
                  >
                    <div className="text-center mb-8">
                      <p className="font-mono text-xs text-white/50 uppercase tracking-widest">Track category spending against your budget.</p>
                    </div>
                    <div className="w-full aspect-video max-w-5xl mx-auto bg-black/40 backdrop-blur-3xl rounded-[3rem] border border-white/10 p-8 relative overflow-hidden shadow-2xl flex gap-6 text-left">
                      <div className="flex-1 bg-white/5 border-[0.5px] border-white/10 rounded-2xl p-6 flex flex-col gap-6">
                        <span className="text-[9px] font-black text-foreground/40 uppercase tracking-[0.3em] border-b border-white/10 pb-4">Category Variance</span>
                        <div className="flex flex-col gap-6">
                          {[
                            { cat: "Groceries", spent: 450, limit: 500, status: "WARNING" },
                            { cat: "Transport", spent: 120, limit: 200, status: "GOOD" },
                            { cat: "Entertainment", spent: 300, limit: 150, status: "EXCEEDED" },
                          ].map((item, i) => (
                            <div key={i} className="flex flex-col gap-2">
                              <div className="flex justify-between items-end">
                                <span className="font-mono text-sm">{item.cat}</span>
                                <span className="font-mono text-xs text-white/50">£{item.spent} / £{item.limit}</span>
                              </div>
                              <div className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden">
                                <div
                                  className={`h-full rounded-full ${item.status === 'EXCEEDED' ? 'bg-rose-500' : item.status === 'WARNING' ? 'bg-amber-500' : 'bg-[#00f2ff]'}`}
                                  style={{ width: `${Math.min((item.spent / item.limit) * 100, 100)}%` }}
                                />
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                      <div className="w-1/3 flex flex-col gap-6">
                        <div className="flex-1 bg-white/5 border-[0.5px] border-white/10 rounded-2xl p-6 flex flex-col justify-center gap-2">
                          <span className="text-[9px] font-black text-foreground/40 uppercase tracking-[0.3em]">Total Discretionary</span>
                          <span className="text-3xl font-mono text-[#00f2ff]">£420 Left</span>
                        </div>
                        <div className="flex-1 bg-rose-500/10 border-[0.5px] border-rose-500/20 rounded-2xl p-6 flex flex-col justify-center gap-2">
                          <span className="text-[9px] font-black text-rose-500/70 uppercase tracking-[0.3em]">Variance Alert</span>
                          <span className="text-sm font-mono text-rose-500 tracking-tight">Entertainment budget exceeded by 100% based on recent trends.</span>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}

                {/* Tab 5: Scenarios */}
                {activeTab === "stresstest" && (
                  <motion.div
                    key="stresstest"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3 }}
                    className="outline-none w-full"
                  >
                    <div className="text-center mb-8">
                      <p className="font-mono text-xs text-white/50 uppercase tracking-widest">Simulate how changes in the economy affect your finances.</p>
                    </div>
                    <div className="w-full aspect-video max-w-5xl mx-auto bg-black/40 backdrop-blur-3xl rounded-[3rem] border border-white/10 p-8 relative overflow-hidden shadow-2xl flex flex-col gap-6 text-left">
                      <div className="flex justify-between items-center bg-white/5 border-[0.5px] border-white/10 p-4 rounded-2xl">
                        <div className="flex gap-4">
                          <div className="px-3 py-1 bg-white/10 rounded text-[9px] font-mono uppercase tracking-widest">Iterations: 10,000</div>
                          <div className="px-3 py-1 bg-white/10 rounded text-[9px] font-mono uppercase tracking-widest">Shock Model: Inflation Spike</div>
                        </div>
                        <div className="px-3 py-1 bg-[#00f2ff]/10 text-[#00f2ff] rounded text-[9px] font-mono uppercase tracking-widest">Resilience Score: 84/100</div>
                      </div>
                      <div className="flex-1 bg-white/5 border-[0.5px] border-white/10 rounded-2xl p-6 relative overflow-hidden">
                        <div className="absolute top-1/2 w-full h-[1px] bg-white/20 border-t border-dashed z-0 left-0" />
                        <span className="absolute top-[calc(50%-16px)] left-6 text-[9px] font-mono text-white/50 uppercase">Base Case Trajectory</span>

                        {/* Simulated overlapping sine/random curves */}
                        <div className="absolute inset-0 flex items-center justify-center opacity-30">
                          <svg viewBox="0 0 100 50" preserveAspectRatio="none" className="w-full h-full stroke-rose-500/50 fill-none" strokeWidth="0.2">
                            <path d="M0,25 Q10,10 20,25 T40,20 T60,30 T80,15 T100,25" />
                            <path d="M0,25 Q15,40 30,25 T50,15 T70,35 T90,20 T100,25" className="stroke-[#00f2ff]/50" />
                            <path d="M0,25 Q20,20 40,30 T60,10 T80,40 T100,25" className="stroke-amber-500/50" />
                          </svg>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}

                {/* Tab 6: Benchmarks */}
                {activeTab === "benchmarks" && (
                  <motion.div
                    key="benchmarks"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3 }}
                    className="outline-none w-full"
                  >
                    <div className="text-center mb-8">
                      <p className="font-mono text-xs text-white/50 uppercase tracking-widest">Compare your spending against national averages.</p>
                    </div>
                    <div className="w-full aspect-video max-w-5xl mx-auto bg-black/40 backdrop-blur-3xl rounded-[3rem] border border-white/10 p-8 relative overflow-hidden shadow-2xl flex flex-col gap-6 text-left">
                      <div className="flex-1 flex gap-6">
                        <div className="flex-1 bg-white/5 border-[0.5px] border-white/10 rounded-2xl p-6 flex flex-col gap-4">
                          <span className="text-[9px] font-black text-foreground/40 uppercase tracking-[0.3em] border-b border-white/10 pb-4">Your Profile</span>
                          <div className="flex flex-col gap-4 justify-center h-full">
                            <div className="flex justify-between items-center"><span className="font-mono text-sm">Housing</span><span className="font-mono text-sm text-[#00f2ff]">35%</span></div>
                            <div className="flex justify-between items-center"><span className="font-mono text-sm">Transport</span><span className="font-mono text-sm text-[#00f2ff]">15%</span></div>
                            <div className="flex justify-between items-center"><span className="font-mono text-sm">Groceries</span><span className="font-mono text-sm text-[#00f2ff]">20%</span></div>
                            <div className="flex justify-between items-center"><span className="font-mono text-sm">Savings</span><span className="font-mono text-sm text-rose-500">10%</span></div>
                          </div>
                        </div>
                        <div className="flex-1 bg-white/5 border-[0.5px] border-white/10 rounded-2xl p-6 flex flex-col gap-4 opacity-50">
                          <span className="text-[9px] font-black text-foreground/40 uppercase tracking-[0.3em] border-b border-white/10 pb-4">ONS UK Average (Top 20%)</span>
                          <div className="flex flex-col gap-4 justify-center h-full">
                            <div className="flex justify-between items-center"><span className="font-mono text-sm">Housing</span><span className="font-mono text-sm">28%</span></div>
                            <div className="flex justify-between items-center"><span className="font-mono text-sm">Transport</span><span className="font-mono text-sm">12%</span></div>
                            <div className="flex justify-between items-center"><span className="font-mono text-sm">Groceries</span><span className="font-mono text-sm">15%</span></div>
                            <div className="flex justify-between items-center"><span className="font-mono text-sm">Savings</span><span className="font-mono text-sm">25%</span></div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}

                {/* Tab 7: Goal Modeling */}
                {activeTab === "goals" && (
                  <motion.div
                    key="goals"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3 }}
                    className="outline-none w-full"
                  >
                    <div className="text-center mb-8">
                      <p className="font-mono text-xs text-white/50 uppercase tracking-widest">See how large purchases affect your savings timeline.</p>
                    </div>
                    <div className="w-full aspect-video max-w-5xl mx-auto bg-black/40 backdrop-blur-3xl rounded-[3rem] border border-white/10 p-8 relative overflow-hidden shadow-2xl flex gap-6 text-left">
                      <div className="w-1/3 flex flex-col gap-4">
                        <div className="bg-white/5 border-[0.5px] border-white/10 rounded-2xl p-6 flex flex-col gap-2">
                          <span className="text-[9px] font-black text-foreground/40 uppercase tracking-[0.3em]">Goal Target</span>
                          <span className="text-2xl font-mono">£35,000</span>
                          <span className="text-[10px] font-mono text-white/40">House Deposit</span>
                        </div>
                        <div className="flex-1 bg-white/5 border-[0.5px] border-white/10 rounded-2xl p-6 flex flex-col gap-4">
                          <span className="text-[9px] font-black text-foreground/40 uppercase tracking-[0.3em]">Adjust Variables</span>
                          <div className="w-full h-1 bg-white/10 rounded"><div className="w-1/2 h-full bg-[#00f2ff]" /></div>
                          <span className="text-[9px] font-mono text-white/40">Monthly Saving: £800</span>
                        </div>
                      </div>
                      <div className="flex-1 bg-white/5 border-[0.5px] border-white/10 rounded-2xl p-6 flex flex-col justify-center items-center text-center">
                        <Activity size={48} className="text-[#00f2ff] opacity-50 mb-4" />
                        <h3 className="text-3xl font-mono tracking-tight">ETA: Aug 2028</h3>
                        <p className="text-sm font-mono text-white/40 mt-2 max-w-sm">If current spending continues, you will hit this target in 24 months.</p>
                      </div>
                    </div>
                  </motion.div>
                )}

                {/* Tab 8: Asset Costs */}
                {activeTab === "tco" && (
                  <motion.div
                    key="tco"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3 }}
                    className="outline-none w-full"
                  >
                    <div className="text-center mb-8">
                      <p className="font-mono text-xs text-white/50 uppercase tracking-widest">Calculate the total cost of owning an asset, including depreciation and maintenance.</p>
                    </div>
                    <div className="w-full aspect-video max-w-5xl mx-auto bg-black/40 backdrop-blur-3xl rounded-[3rem] border border-white/10 p-8 relative overflow-hidden shadow-2xl flex flex-col gap-6 text-left">
                      <div className="flex justify-between items-center bg-white/5 border-[0.5px] border-white/10 p-4 rounded-2xl">
                        <span className="font-mono font-bold tracking-widest uppercase">Asset: Tesla Model 3 (2025)</span>
                        <span className="font-mono text-[#00f2ff]">Sticker Price: £42,000</span>
                      </div>
                      <div className="flex-1 flex gap-6">
                        <div className="flex-1 flex flex-col gap-2 justify-end bg-white/5 rounded-2xl p-6 border-[0.5px] border-white/10">
                          <div className="w-full bg-[#7000ff]/20 border-t border-[#7000ff]/50 flex items-center justify-center text-[10px] font-mono text-[#7000ff]" style={{ height: '60%' }}>£12k Depr.</div>
                          <div className="w-full bg-rose-500/20 border-t border-rose-500/50 flex items-center justify-center text-[10px] font-mono text-rose-500" style={{ height: '15%' }}>£3k Ins.</div>
                          <div className="w-full bg-[#00f2ff]/20 border-t border-[#00f2ff]/50 flex items-center justify-center text-[10px] font-mono text-[#00f2ff]" style={{ height: '25%' }}>£5k Maint.</div>
                        </div>
                        <div className="flex-1 bg-white/5 border-[0.5px] border-white/10 rounded-2xl p-6 flex flex-col justify-center">
                          <span className="text-[9px] font-black text-foreground/40 uppercase tracking-[0.3em]">True 5-Year Cost</span>
                          <span className="text-5xl font-mono text-foreground tracking-tighter mt-2">£62,000</span>
                          <span className="text-xs font-mono text-rose-500 mt-4 border-l-2 border-rose-500 pl-2">That is £333/mo more than your current surplus.</span>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

          </div>
        </section>

        {/* Data Pipeline Diagram */}
        <section className="py-24 bg-background border-t border-white/5 relative z-20 overflow-hidden">
          <div className="max-w-7xl mx-auto px-6 relative">
            <div className="text-center mb-16">
              <h2 className="text-4xl md:text-5xl font-mono font-black tracking-tighter mb-6">
                DATA PIPELINE
              </h2>
              <h3 className="font-mono text-sm font-black text-white/40 tracking-[0.2em] uppercase">
                How Your Data Flows
              </h3>
            </div>
            {/* Track (Desktop) */}
            <div className="relative w-full h-[320px] hidden md:block mt-8">
              {/* Background Lines */}
              {/* Main horizontal line (Center Y: 24px) */}
              <div className="absolute top-[24px] left-[8.33%] right-[8.33%] h-[1px] bg-white/10 z-0" />
              {/* AI horizontal line (Center Y: 164px) */}
              <div className="absolute top-[164px] left-[25%] right-[8.33%] h-[1px] bg-white/10 z-0" />
              {/* Vertical line from Storage (Col 2: 25%) */}
              <div className="absolute top-[24px] left-[25%] w-[1px] h-[140px] bg-white/10 z-0" />
              {/* Vertical line to Overview (Col 6: 91.66%) */}
              <div className="absolute top-[24px] right-[8.33%] w-[1px] h-[140px] bg-white/10 z-0" />

              {/* Moving Packets (Data) - Main Track */}
              {[0, 1.0, 2.0, 3.0, 4.0].map((delay, i) => (
                <motion.div
                  key={`main-packet-${i}`}
                  animate={{
                    left: ["8.33%", "91.66%"],
                    opacity: [0, 1, 1, 0]
                  }}
                  transition={{
                    duration: 5,
                    repeat: Infinity,
                    ease: "linear",
                    delay,
                    times: [0, 0.1, 0.9, 1]
                  }}
                  className="absolute top-[20px] w-12 h-2 -ml-11 z-10 flex items-center justify-end"
                >
                  <div className="w-10 h-[1px] bg-gradient-to-r from-transparent to-[#00f2ff]/80 mr-1" />
                  <div className="w-2 h-2 rounded-full bg-[#00f2ff] shadow-[0_0_10px_#00f2ff]" />
                </motion.div>
              ))}

              {/* Moving Packets (Data) - AI Track */}
              {[0, 1.0, 2.0, 3.0, 4.0, 5.0].map((delay, i) => (
                <motion.div
                  key={`ai-packet-${i}`}
                  animate={{
                    left: ["25%", "25%", "91.66%", "91.66%"],
                    top: ["20px", "160px", "160px", "20px"],
                    opacity: [0, 1, 1, 0]
                  }}
                  transition={{
                    duration: 6,
                    repeat: Infinity,
                    ease: "linear",
                    delay: delay + 1,
                    times: [0, 0.15, 0.85, 1]
                  }}
                  className="absolute w-2 h-2 -ml-[4px] rounded-full bg-[#7000ff] shadow-[0_0_10px_#7000ff] z-10"
                />
              ))}

              {/* Main Checkpoints */}
              {[
                { icon: Lock, title: "Bank Connection", desc: "Syncs Data", left: "8.33%", delay: 0.5 },
                { icon: Database, title: "Storage", desc: "Saves Data", left: "25%", delay: 0.3 },
                { icon: Activity, title: "Review", desc: "Checks Merchants", left: "41.66%", delay: 0.1 },
                { icon: Search, title: "Research", desc: "Searches Web", left: "58.33%", delay: 0.9 },
                { icon: Activity, title: "Planning", desc: "Checks Subscriptions", left: "75%", delay: 0.7 },
                { icon: TrendingUp, title: "Overview", desc: "Shows Balances", left: "91.66%", delay: 0.5 },
              ].map((step, i) => (
                <div key={i} className="absolute top-0 w-32 -ml-16 flex flex-col items-center gap-4 group z-20" style={{ left: step.left }}>
                  <div className="relative">
                    {i === 5 && (
                      <motion.div
                        animate={{ scale: [1, 2.5], opacity: [0.8, 0] }}
                        transition={{ duration: 1.0, repeat: Infinity, delay: 0.5 }}
                        className="absolute inset-0 rounded-full border border-[#00f2ff] z-10 pointer-events-none"
                      />
                    )}
                    <motion.div
                      animate={{
                        borderColor: ["rgba(255,255,255,0.2)", "rgba(0,242,255,0.8)", "rgba(255,255,255,0.2)"],
                        boxShadow: ["0 0 0px rgba(0,0,0,0)", "0 0 20px rgba(0,242,255,0.5)", "0 0 0px rgba(0,0,0,0)"],
                        color: ["rgba(255,255,255,0.5)", "rgba(0,242,255,1)", "rgba(255,255,255,0.5)"]
                      }}
                      transition={{
                        duration: 1.0,
                        repeat: Infinity,
                        delay: step.delay,
                        times: [0, 0.2, 1]
                      }}
                      className="w-12 h-12 bg-background rounded-full border border-white/20 flex items-center justify-center text-white/50 z-20 relative"
                    >
                      <step.icon size={18} strokeWidth={1.5} />
                    </motion.div>
                  </div>
                  <div className="text-center px-2">
                    <div className="font-mono text-white text-[10px] uppercase tracking-widest font-black mb-1">{step.title}</div>
                    <div className="font-mono text-[8px] text-white/40 uppercase tracking-widest">{step.desc}</div>
                  </div>
                </div>
              ))}

              {/* AI Checkpoints */}
              {[
                { icon: Database, title: "Memory Bank", desc: "Recalls History", left: "41.66%", delay: 0.95 },
                { icon: Sparkles, title: "Smart Assistant", desc: "Finds Patterns", left: "58.33%", delay: 0.0 },
                { icon: MessageSquare, title: "Smart Insights", desc: "Provides Advice", left: "75%", delay: 0.05 },
              ].map((step, i) => (
                <div key={i} className="absolute top-[140px] w-32 -ml-16 flex flex-col items-center gap-4 group z-20" style={{ left: step.left }}>
                  <motion.div
                    animate={{
                      borderColor: ["rgba(255,255,255,0.2)", "rgba(112,0,255,0.8)", "rgba(255,255,255,0.2)"],
                      boxShadow: ["0 0 0px rgba(0,0,0,0)", "0 0 20px rgba(112,0,255,0.5)", "0 0 0px rgba(0,0,0,0)"],
                      color: ["rgba(112,0,255,0.5)", "rgba(112,0,255,1)", "rgba(112,0,255,0.5)"]
                    }}
                    transition={{
                      duration: 1.0,
                      repeat: Infinity,
                      delay: step.delay,
                      times: [0, 0.2, 1]
                    }}
                    className="w-12 h-12 bg-background rounded-full border border-white/20 flex items-center justify-center text-[#7000ff] z-20 relative"
                  >
                    <step.icon size={18} strokeWidth={1.5} />
                  </motion.div>
                  <div className="text-center px-2">
                    <div className="font-mono text-white text-[10px] uppercase tracking-widest font-black mb-1">{step.title}</div>
                    <div className="font-mono text-[8px] text-white/40 uppercase tracking-widest">{step.desc}</div>
                  </div>
                </div>
              ))}
            </div>

            {/* Mobile Fallback - Vertical List */}
            <div className="flex flex-col md:hidden">
              <div className="border-l border-white/10 ml-6 pl-8 py-4 relative flex flex-col gap-8">
                {[
                  { icon: Lock, title: "Bank Connection", desc: "Syncs Data" },
                  { icon: Database, title: "Storage", desc: "Saves Data" },
                  { icon: Activity, title: "Review", desc: "Checks Merchants" },
                  { icon: Search, title: "Research", desc: "Searches Web" },
                  { icon: Activity, title: "Planning", desc: "Checks Subscriptions" },
                ].map((step, i) => (
                  <div key={i} className="flex items-center gap-6 relative">
                    <div className="absolute -left-[3.25rem] w-12 h-12 rounded-full bg-[#0c131d] border border-white/20 flex items-center justify-center text-[#00f2ff] shadow-[0_0_15px_rgba(0,242,255,0.1)] flex-shrink-0">
                      <step.icon size={18} strokeWidth={1.5} />
                    </div>
                    <div className="flex flex-col">
                      <div className="font-mono text-white text-[11px] uppercase tracking-widest font-black">{step.title}</div>
                      <div className="font-mono text-[9px] text-white/40 uppercase tracking-widest">{step.desc}</div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="font-mono text-[10px] text-[#7000ff] tracking-widest font-black uppercase mt-4 mb-2 pl-2">AI Processing Track</div>

              <div className="border-l border-[#7000ff]/30 ml-6 pl-8 py-4 relative flex flex-col gap-8 mb-4">
                {[
                  { icon: Database, title: "Memory Bank", desc: "Recalls History" },
                  { icon: Sparkles, title: "Smart Assistant", desc: "Finds Patterns" },
                  { icon: MessageSquare, title: "Smart Insights", desc: "Provides Advice" },
                ].map((step, i) => (
                  <div key={i} className="flex items-center gap-6 relative">
                    <div className="absolute -left-[3.25rem] w-12 h-12 rounded-full bg-[#0c131d] border border-[#7000ff]/30 flex items-center justify-center text-[#7000ff] shadow-[0_0_15px_rgba(112,0,255,0.1)] flex-shrink-0">
                      <step.icon size={18} strokeWidth={1.5} />
                    </div>
                    <div className="flex flex-col">
                      <div className="font-mono text-white text-[11px] uppercase tracking-widest font-black">{step.title}</div>
                      <div className="font-mono text-[9px] text-white/40 uppercase tracking-widest">{step.desc}</div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="flex items-center gap-6 relative ml-6">
                <div className="absolute -left-[1.25rem] w-12 h-12 rounded-full bg-[#0c131d] border border-white/20 flex items-center justify-center text-[#00f2ff] shadow-[0_0_15px_rgba(0,242,255,0.1)] flex-shrink-0">
                  <TrendingUp size={18} strokeWidth={1.5} />
                </div>
                <div className="flex flex-col pl-8">
                  <div className="font-mono text-white text-[11px] uppercase tracking-widest font-black">Overview</div>
                  <div className="font-mono text-[9px] text-white/40 uppercase tracking-widest">Shows Balances</div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section
          id="features"
          className="py-40 px-6 relative z-20 bg-white/2 backdrop-blur-3xl border-y-[0.5px] border-white/5"
        >
          <div className="max-w-6xl mx-auto">
            <div className="text-center mb-24">
              <motion.h2
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                className="text-4xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-linear-to-r from-foreground to-foreground/50 mb-6 tracking-tight leading-tight"
              >
                System{" "}
                <span className="text-transparent bg-clip-text bg-linear-to-r from-[#7000ff] to-[#00f2ff] drop-shadow-[0_0_15px_rgba(112,0,255,0.3)]">
                  Architecture
                </span>
              </motion.h2>
              <p className="text-[10px] font-black text-[#00f2ff] uppercase tracking-[0.4em] mb-4 drop-shadow-md">
                Platform Capabilities
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-10 relative">
              <div className="hidden md:block absolute top-1/2 left-0 w-full h-[0.5px] bg-linear-to-r from-transparent via-[#7000ff]/30 to-transparent -translate-y-1/2 z-0" />

              {[
                {
                  step: "01",
                  title: "Secure Connection",
                  icon: Database,
                  desc: "Bank-level, read-only access to your financial data via secure APIs.",
                },
                {
                  step: "02",
                  title: "Transaction Analysis",
                  icon: MessageSquare,
                  desc: "Automated categorization mapping every transaction with precision.",
                },
                {
                  step: "03",
                  title: "Financial Projections",
                  icon: Sparkles,
                  desc: "Forecast cash flow and analyze spending patterns.",
                },
              ].map((item, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{
                    duration: 0.6,
                    delay: i * 0.2,
                    ease: [0.16, 1, 0.3, 1] as const,
                  }}
                  whileHover={{ y: -5, scale: 1.01 }}
                  className="bg-black/40 backdrop-blur-xl border-[0.5px] border-white/10 rounded-2xl p-12 relative z-10 flex flex-col gap-10 transition-all group shadow-inner"
                >
                  <div className="font-mono text-[#7000ff] text-6xl font-black opacity-10 absolute -top-4 -right-4 italic pointer-events-none group-hover:opacity-20 transition-opacity">
                    {item.step}
                  </div>
                  <div className="w-14 h-14 rounded-xl bg-white/5 border-[0.5px] border-white/10 flex items-center justify-center text-[#00f2ff] shadow-[0_0_20px_rgba(0,242,255,0.05)] group-hover:border-[#00f2ff]/50 transition-all">
                    <item.icon size={28} />
                  </div>
                  <div className="space-y-4">
                    <h3 className="text-xl font-bold tracking-wide text-foreground">
                      {item.title}
                    </h3>
                    <p className="text-xs text-foreground/40 leading-relaxed font-medium tracking-wide">
                      {item.desc}
                    </p>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        <section id="security" className="py-40 px-6 relative z-20">
          <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            className="max-w-6xl mx-auto bg-black/40 backdrop-blur-3xl rounded-[4rem] p-12 md:p-24 border border-white/10 flex flex-col md:flex-row gap-20 items-center shadow-2xl relative overflow-hidden"
          >
            <div className="absolute top-0 right-0 w-96 h-96 bg-primary/5 blur-[120px] rounded-full pointer-events-none" />

            <div className="flex-1 flex flex-col gap-10 text-center md:text-left items-center md:items-start relative z-10">
              <h2 className="text-4xl md:text-5xl font-extrabold text-white tracking-tight leading-tight drop-shadow-md">
                Data{" "}
                <span className="text-transparent bg-clip-text bg-linear-to-r from-[#7000ff] to-[#00f2ff]">
                  Integrity
                </span>
              </h2>
              <p className="text-lg text-foreground/60 max-w-xl font-medium tracking-wide leading-relaxed">
                Secure read-only API access to your financial institutions with
                automated categorization.
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-10 w-full">
                <div className="flex items-start gap-5">
                  <div className="bg-primary/10 text-primary border border-primary/20 p-3 rounded-2xl ">
                    <Lock size={24} />
                  </div>
                  <div>
                    <h4 className="font-black tracking-tight text-foreground uppercase text-sm mb-1">
                      Secure Connection
                    </h4>
                    <p className="text-xs text-foreground/40 font-medium tracking-wide">
                      Read-only access via encrypted endpoints.
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-5">
                  <div className="bg-primary/10 text-primary border border-primary/20 p-3 rounded-2xl ">
                    <EyeOff size={24} />
                  </div>
                  <div>
                    <h4 className="font-black tracking-tight text-foreground uppercase text-sm mb-1">
                      Data Privacy
                    </h4>
                    <p className="text-xs text-foreground/40 font-medium tracking-wide">
                      Your data is never sold or shared.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="w-full md:w-1/3 flex justify-center relative z-10">
              <div className="relative w-56 h-56 rounded-full bg-white/5 border border-white/10 backdrop-blur-2xl flex items-center justify-center shadow-inner">
                <div className="text-center">
                  <Shield className="w-16 h-16 text-[#00f2ff] mx-auto opacity-60" />
                </div>
              </div>
            </div>
          </motion.div>
        </section>

        <div className="w-full bg-background py-12 border-y-[0.5px] border-white/5 flex z-30 relative justify-center">
          <div className="flex flex-wrap justify-center gap-12 md:gap-16 items-center text-xs font-semibold text-foreground/60 max-w-6xl px-6">
            {[
              "Read-Only API Access",
              "End-to-End Encryption",
              "Automated Categorization",
              "Cash Flow Optimization",
              "Multi-Factor Security",
            ].map((stat, i) => (
              <span key={i} className="flex items-center gap-3">
                <span className="w-1.5 h-1.5 rounded-full bg-[#00f2ff]/40" />
                {stat}
              </span>
            ))}
          </div>
        </div>

        <section id="faq" className="py-40 px-6 relative z-20">
          <div className="max-w-3xl mx-auto">
            <div className="text-center mb-20">
              <h2 className="text-4xl md:text-5xl font-extrabold text-white tracking-tight leading-tight drop-shadow-md">
                Support{" "}
                <span className="text-transparent bg-clip-text bg-linear-to-r from-[#7000ff] to-[#00f2ff] drop-shadow-[0_0_15px_rgba(0,242,255,0.3)]">
                  Center
                </span>
              </h2>
            </div>

            <div className="flex flex-col gap-6">
              {[
                {
                  id: "1",
                  title: "Data Security & Privacy",
                  content:
                    "We maintain the highest standard of read-only access. Your money remains entirely under your control.",
                },
                {
                  id: "2",
                  title: "Student & Professional Tools",
                  content:
                    "Designed to track complex spending habits—from student loans to professional income.",
                },
                {
                  id: "3",
                  title: "Automated Insights",
                  content:
                    "Once connected, our platform operates continuously to categorize your transactions without manual input.",
                },
              ].map((faq) => (
                <Accordion
                  key={faq.id}
                  className="bg-black/40 backdrop-blur-xl border-[0.5px] border-white/10 rounded-2xl overflow-hidden shadow-inner"
                >
                  <Accordion.Item>
                    <Accordion.Heading>
                      <Accordion.Trigger className="w-full p-8 font-black text-foreground uppercase tracking-[0.2em] text-[10px] flex justify-between items-center hover:bg-white/5 transition-all">
                        {faq.title}
                        <ChevronDown size={18} className="text-primary" />
                      </Accordion.Trigger>
                    </Accordion.Heading>
                    <Accordion.Panel className="px-8 pb-8 text-foreground/40 font-medium tracking-wide text-xs leading-relaxed">
                      <div className="pt-6 border-t-[0.5px] border-white/5">
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

      <footer className="border-t-[0.5px] border-white/5 bg-black/40 backdrop-blur-3xl pt-32 pb-16 px-6 relative z-20">
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-20 mb-24">
            <div className="lg:col-span-2 flex flex-col gap-10">
              <div className="flex items-center gap-3">
                <Image
                  src="/FullLogo.jpg"
                  alt="BudAI Logo"
                  width={340}
                  height={112}
                  className="h-28 w-auto rounded-md object-contain"
                />
              </div>
              <p className="text-foreground/30 max-w-xs leading-relaxed font-medium tracking-wide text-xs">
                BudAI. Personal Financial Advisor. Securely connect your bank to
                analyze spending and predict future balances using quantitative
                calculations.
              </p>
            </div>

            <div className="flex flex-col gap-8">
              <h4 className="text-foreground font-black text-[10px] uppercase tracking-[0.4em] opacity-40">
                Core
              </h4>
              <nav className="flex flex-col gap-4">
                <Link
                  href="#"
                  className="text-foreground/30 hover:text-primary transition-all text-[10px] font-bold uppercase tracking-widest"
                >
                  Features
                </Link>
                <Link
                  href="#"
                  className="text-foreground/30 hover:text-primary transition-all text-[10px] font-bold uppercase tracking-widest"
                >
                  Projections
                </Link>
                <Link
                  href="#"
                  className="text-foreground/30 hover:text-primary transition-all text-[10px] font-bold uppercase tracking-widest"
                >
                  Security
                </Link>
                <Link
                  href="#"
                  className="text-foreground/30 hover:text-primary transition-all text-[10px] font-bold uppercase tracking-widest"
                >
                  Pricing
                </Link>
              </nav>
            </div>

            <div className="flex flex-col gap-8">
              <h4 className="text-foreground font-black text-[10px] uppercase tracking-[0.4em] opacity-40">
                Resources
              </h4>
              <nav className="flex flex-col gap-4">
                <Link
                  href="#"
                  className="text-foreground/30 hover:text-primary transition-all text-[10px] font-bold uppercase tracking-widest"
                >
                  Documentation
                </Link>
                <Link
                  href="#"
                  className="text-foreground/30 hover:text-primary transition-all text-[10px] font-bold uppercase tracking-widest"
                >
                  API Access
                </Link>
                <Link
                  href="#"
                  className="text-foreground/30 hover:text-primary transition-all text-[10px] font-bold uppercase tracking-widest"
                >
                  Status
                </Link>
                <Link
                  href="#"
                  className="text-foreground/30 hover:text-primary transition-all text-[10px] font-bold uppercase tracking-widest"
                >
                  Blog
                </Link>
              </nav>
            </div>

            <div className="flex flex-col gap-8">
              <h4 className="text-foreground font-black text-[10px] uppercase tracking-[0.4em] opacity-40">
                Legal
              </h4>
              <nav className="flex flex-col gap-4">
                <Link
                  href="#"
                  className="text-foreground/30 hover:text-primary transition-all text-[10px] font-bold uppercase tracking-widest"
                >
                  Privacy
                </Link>
                <Link
                  href="#"
                  className="text-foreground/30 hover:text-primary transition-all text-[10px] font-bold uppercase tracking-widest"
                >
                  Terms
                </Link>
                <Link
                  href="#"
                  className="text-foreground/30 hover:text-primary transition-all text-[10px] font-bold uppercase tracking-widest"
                >
                  Cookies
                </Link>
              </nav>
            </div>
          </div>

          <div className="flex flex-col md:flex-row justify-between items-center pt-10 border-t-[0.5px] border-white/5 gap-8 text-foreground/20 text-[9px] font-black uppercase tracking-[0.3em]">
            <p>© 2026 BudAI Systems. All rights reserved.</p>
            <div className="flex gap-12">
              <Link
                href="#"
                className="text-inherit hover:text-foreground transition-all"
              >
                X / Twitter
              </Link>
              <Link
                href="#"
                className="text-inherit hover:text-foreground transition-all"
              >
                LinkedIn
              </Link>
              <Link
                href="#"
                className="text-inherit hover:text-foreground transition-all"
              >
                GitHub
              </Link>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
