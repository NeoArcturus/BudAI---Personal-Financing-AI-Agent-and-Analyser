"use client";

import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Button, Link, Accordion } from "@heroui/react";
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
  BrainCircuit,
} from "lucide-react";
import { motion } from "framer-motion";

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
    <div className="min-h-screen text-foreground selection:bg-primary/30 selection:text-primary relative overflow-x-hidden bg-transparent font-sans">
      <nav className="fixed top-0 w-full z-50 h-16 bg-black/20 backdrop-blur-xl border-b-[0.5px] border-white/5 flex justify-between items-center px-6 md:px-10 transition-all">
        <div className="flex items-center gap-2">
          <Activity className="text-primary w-6 h-6" />
          <span className="font-bold text-foreground text-lg tracking-tighter uppercase italic">
            BudAI
          </span>
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
            href="#engine"
          >
            Engine
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
            onPress={() => router.push("/register")}
            className="bg-primary text-primary-foreground font-black text-[10px] uppercase tracking-widest px-6 h-9 rounded-md shadow-[0_0_15px_rgba(0,127,255,0.2)] hover:shadow-[0_0_25px_rgba(0,127,255,0.4)] transition-all border-none"
          >
            Sign Up
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

      <main className="relative z-10 pt-16">
        <section className="relative min-h-[90vh] flex items-center justify-center px-6 py-24">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="max-w-4xl mx-auto text-center flex flex-col items-center gap-10"
          >
            <motion.div
              variants={itemVariants}
              className="inline-flex items-center gap-3 px-4 py-1.5 rounded-lg border-[0.5px] border-primary/30 bg-primary/5 text-primary text-[9px] font-black uppercase tracking-[0.3em] shadow-[0_0_15px_rgba(0,127,255,0.05)]"
            >
              <div className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
              <span>All Systems Operational</span>
            </motion.div>

            <motion.h1
              variants={itemVariants}
              className="text-5xl md:text-8xl font-normal tracking-tighter text-foreground leading-[0.85] uppercase italic"
            >
              Master Your <br />
              <span className="text-transparent bg-clip-text bg-linear-to-r from-primary via-primary/80 to-primary/40 not-italic font-black">
                Money with AI
              </span>
            </motion.h1>

            <motion.p
              variants={itemVariants}
              className="text-base md:text-lg text-foreground/50 max-w-xl mx-auto leading-relaxed font-medium tracking-tight"
            >
              The most precise personal finance companion for students and
              professionals. Achieve complete clarity and hit goals faster.
            </motion.p>

            <motion.div
              variants={itemVariants}
              className="flex flex-col sm:flex-row items-center gap-6 mt-4 w-full sm:w-auto"
            >
              <Button
                onPress={() => router.push("/register")}
                className="bg-primary text-primary-foreground font-black uppercase tracking-[0.2em] text-[11px] px-12 h-14 rounded-xl shadow-[0_0_30px_rgba(0,127,255,0.3)] hover:shadow-[0_0_40px_rgba(0,127,255,0.5)] hover:scale-[1.02] transition-all border-none"
              >
                Go to Dashboard
              </Button>
              <Button
                variant="outline"
                className="border-white/10 text-foreground/60 px-12 h-14 text-[11px] font-bold uppercase tracking-[0.2em] rounded-xl bg-white/5 backdrop-blur-xl hover:bg-white/10 hover:text-foreground transition-all"
              >
                Learn More
              </Button>
            </motion.div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 0.6, y: 0 }}
            transition={{ duration: 1.2, delay: 0.8, ease: [0.16, 1, 0.3, 1] }}
            className="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-1/2 w-full max-w-5xl h-64 bg-black/40 backdrop-blur-3xl rounded-t-[3rem] border-t-[0.5px] border-x-[0.5px] border-white/10 flex justify-center pt-10 overflow-hidden pointer-events-none"
          >
            <motion.div
              animate={{ y: [0, -10, 0] }}
              transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
              className="w-4/5 h-full border-[0.5px] border-white/10 rounded-t-2xl bg-white/2 flex p-8 gap-8 shadow-inner"
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

        <section className="py-24 bg-black/10 backdrop-blur-md border-y border-white/5 overflow-hidden relative z-20">
          <div className="text-center mb-12">
            <h3 className="text-[10px] font-black text-primary/40 uppercase tracking-[0.4em]">
              Securely connect to your institutions
            </h3>
          </div>
          <div className="relative flex overflow-x-hidden w-full">
            <div className="flex animate-marquee whitespace-nowrap gap-24 items-center py-6 opacity-30 px-12">
              {[
                "HSBC",
                "BARCLAYS",
                "MONZO",
                "REVOLUT",
                "LLOYDS",
                "CHASE",
                "STARLING",
                "NATIONWIDE",
                "NATWEST",
              ].map((bank, i) => (
                <span
                  key={i}
                  className="text-4xl font-black text-foreground tracking-tighter italic"
                >
                  {bank}
                </span>
              ))}
            </div>
          </div>
        </section>

        <section id="engine" className="py-40 px-6 relative z-20">
          <div className="max-w-6xl mx-auto flex flex-col items-center">
            <div className="text-center mb-24">
              <motion.h2
                initial={{ opacity: 0, scale: 0.95 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                className="text-4xl md:text-6xl font-black text-foreground mb-6 tracking-tighter uppercase italic"
              >
                Smart Insights, <br />
                Simple Actions
              </motion.h2>
              <p className="text-lg text-foreground/60 max-w-2xl mx-auto font-medium tracking-wide">
                Your money tells a story. BudAI helps you read it, offering
                clear advice to grow your savings and spend smarter every day.
              </p>
            </div>

            <div className="w-full aspect-video max-w-5xl bg-black/40 backdrop-blur-3xl rounded-[3rem] border border-white/10 p-6 relative overflow-hidden group shadow-2xl">
              <div className="w-full h-full bg-white/5 rounded-[2rem] flex border border-white/5 overflow-hidden">
                <div className="w-1/4 border-r border-white/5 p-8 hidden md:flex flex-col gap-8">
                  <div className="h-10 bg-white/10 rounded-xl w-3/4 mb-4" />
                  <div className="h-5 bg-white/5 rounded-lg w-full" />
                  <div className="h-5 bg-white/5 rounded-lg w-5/6" />
                  <div className="h-5 bg-white/5 rounded-lg w-full" />
                </div>

                <div className="flex-1 p-10 flex flex-col gap-10 relative">
                  <motion.div
                    animate={{ y: [0, -8, 0] }}
                    transition={{
                      duration: 4,
                      repeat: Infinity,
                      ease: "easeInOut",
                    }}
                    className="absolute top-1/4 left-1/4 bg-primary/20 border border-primary/30 text-primary px-5 py-2.5 rounded-full text-xs font-black uppercase tracking-widest flex items-center gap-3 shadow-xl z-10 hover:scale-105 transition-transform cursor-default"
                  >
                    <TrendingUp size={18} /> Budget Performance +12.4%
                  </motion.div>
                  <motion.div
                    animate={{ y: [0, 8, 0] }}
                    transition={{
                      duration: 5,
                      repeat: Infinity,
                      ease: "easeInOut",
                      delay: 1,
                    }}
                    className="absolute bottom-1/3 right-1/4 bg-pink-500/20 border border-pink-500/30 text-pink-500 px-5 py-2.5 rounded-full text-xs font-black uppercase tracking-widest flex items-center gap-3 shadow-xl z-10 hover:scale-105 transition-transform cursor-default"
                  >
                    <Activity size={18} /> Subscription Alert: Netflix
                  </motion.div>

                  <div className="grid grid-cols-3 gap-8">
                    <div className="h-40 bg-white/5 rounded-3xl border border-white/5 animate-pulse" />
                    <div className="h-40 bg-white/5 rounded-3xl border border-white/5" />
                    <div className="h-40 bg-white/5 rounded-3xl border border-white/5" />
                  </div>

                  <div className="flex-1 bg-white/5 rounded-3xl border border-white/5 p-8">
                    <div className="w-full h-full border-b border-l border-white/10 flex items-end p-6 gap-6">
                      {[25, 50, 35, 75, 90, 60, 45].map((h, i) => (
                        <motion.div
                          key={i}
                          initial={{ height: 0 }}
                          whileInView={{ height: `${h}%` }}
                          viewport={{ once: true }}
                          transition={{
                            duration: 1.5,
                            delay: i * 0.1,
                            ease: "easeOut",
                          }}
                          className="flex-1 bg-primary/30 rounded-t-xl border-t border-x border-primary/20 hover:bg-primary/50 transition-colors cursor-pointer"
                        />
                      ))}
                    </div>
                  </div>
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
                className="text-4xl md:text-6xl font-normal text-foreground mb-6 tracking-tighter uppercase italic"
              >
                How BudAI <br />
                <span className="font-black not-italic">Works</span>
              </motion.h2>
              <p className="text-[10px] font-black text-primary uppercase tracking-[0.4em] mb-4">
                Core System Architecture
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-10 relative">
              <div className="hidden md:block absolute top-1/2 left-0 w-full h-[0.5px] bg-linear-to-r from-transparent via-primary/20 to-transparent -translate-y-1/2 z-0" />

              {[
                {
                  step: "01",
                  title: "Secure Connection",
                  icon: Database,
                  desc: "Bank-level, read-only access to your financial data via secure APIs.",
                },
                {
                  step: "02",
                  title: "Intelligent Analysis",
                  icon: BrainCircuit,
                  desc: "Automated categorization mapping every transaction with precision.",
                },
                {
                  step: "03",
                  title: "Predictive Insights",
                  icon: Sparkles,
                  desc: "AI-driven forecasting to optimize your cash flow and savings.",
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
                  <div className="font-mono text-primary text-6xl font-black opacity-5 absolute -top-4 -right-4 italic pointer-events-none group-hover:opacity-10 transition-opacity">
                    {item.step}
                  </div>
                  <div className="w-14 h-14 rounded-xl bg-white/5 border-[0.5px] border-white/10 flex items-center justify-center text-primary shadow-[0_0_20px_rgba(0,127,255,0.05)] group-hover:border-primary/50 transition-all">
                    <item.icon size={28} />
                  </div>
                  <div className="space-y-4">
                    <h3 className="text-xl font-black tracking-widest text-foreground uppercase italic">
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
              <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-primary/30 bg-primary/10 text-primary text-[10px] font-black uppercase tracking-[0.2em]">
                <Shield size={14} />
                <span>Your Privacy is our priority</span>
              </div>
              <h2 className="text-4xl md:text-6xl font-black text-foreground tracking-tighter uppercase italic leading-[0.95]">
                Built on Trust, <br />
                Not Technicality
              </h2>
              <p className="text-lg text-foreground/60 max-w-xl font-medium tracking-wide leading-relaxed">
                We believe your financial data belongs to you. We use the same
                protection as major banks, but with a simple promise: we never
                sell your data, and we only have read-only access.
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-10 w-full">
                <div className="flex items-start gap-5">
                  <div className="p-3 rounded-2xl bg-primary/10 text-primary border border-primary/20">
                    <Lock size={24} />
                  </div>
                  <div>
                    <h4 className="font-black tracking-tight text-foreground uppercase text-sm mb-1">
                      Bank-Grade Safety
                    </h4>
                    <p className="text-xs text-foreground/40 font-medium tracking-wide">
                      Top-tier protection for your peace of mind.
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-5">
                  <div className="p-3 rounded-2xl bg-primary/10 text-primary border border-primary/20">
                    <EyeOff size={24} />
                  </div>
                  <div>
                    <h4 className="font-black tracking-tight text-foreground uppercase text-sm mb-1">
                      Pure Privacy
                    </h4>
                    <p className="text-xs text-foreground/40 font-medium tracking-wide">
                      No data selling. No hidden sharing. Ever.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="w-full md:w-1/3 flex justify-center relative z-10">
              <div className="relative w-72 h-72 flex items-center justify-center">
                <motion.svg
                  animate={{ rotate: 360 }}
                  transition={{
                    duration: 20,
                    repeat: Infinity,
                    ease: "linear",
                  }}
                  className="absolute inset-0 w-full h-full"
                >
                  <circle
                    cx="144"
                    cy="144"
                    r="130"
                    fill="none"
                    stroke="rgba(255,255,255,0.05)"
                    strokeWidth="8"
                    strokeDasharray="10 20"
                  />
                  <circle
                    cx="144"
                    cy="144"
                    r="110"
                    fill="none"
                    stroke="rgba(0,127,255,0.2)"
                    strokeWidth="2"
                  />
                </motion.svg>
                <div className="relative w-56 h-56 rounded-full bg-white/5 border border-white/10 backdrop-blur-2xl flex items-center justify-center shadow-inner">
                  <div className="text-center">
                    <Shield className="w-12 h-12 text-primary mx-auto mb-3 drop-shadow-[0_0_15px_rgba(0,127,255,0.6)]" />
                    <span className="block text-2xl font-black text-foreground tracking-tighter uppercase italic">
                      SECURE
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </section>

        <div className="w-full bg-black/10 backdrop-blur-md py-6 border-y-[0.5px] border-white/5 overflow-hidden flex z-30 relative">
          <div className="flex animate-marquee whitespace-nowrap gap-16 items-center text-[9px] font-black text-primary/40 uppercase tracking-[0.4em]">
            {[
              "Read-Only API Access",
              "End-to-End Encryption",
              "Automated Categorization",
              "Cash Flow Optimization",
              "Multi-Factor Security",
            ].map((stat, i) => (
              <span key={i} className="flex items-center gap-4">
                <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse shadow-[0_0_10px_rgba(0,127,255,0.8)]" />
                {stat}
              </span>
            ))}

            {[
              "Read-Only API Access",
              "End-to-End Encryption",
              "Automated Categorization",
              "Cash Flow Optimization",
              "Multi-Factor Security",
            ].map((stat, i) => (
              <span key={`dup-${i}`} className="flex items-center gap-4 ml-16">
                <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse shadow-[0_0_10px_rgba(0,127,255,0.8)]" />
                {stat}
              </span>
            ))}
          </div>
        </div>

        <section id="faq" className="py-40 px-6 relative z-20">
          <div className="max-w-3xl mx-auto">
            <div className="text-center mb-20">
              <h2 className="text-4xl md:text-6xl font-normal text-foreground tracking-tighter uppercase italic">
                Support <span className="font-black not-italic">Center</span>
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
                <Activity className="text-primary w-6 h-6" />
                <span className="text-xl font-black text-foreground tracking-tighter uppercase italic">
                  BudAI
                </span>
              </div>
              <p className="text-foreground/30 max-w-xs leading-relaxed font-medium tracking-wide text-xs">
                BudAI. AI-Driven Personal Finance. Securely connect your bank to
                analyze spending and predict future balances using local AI
                models.
              </p>
              <div className="flex items-center gap-3 text-[9px] font-black text-green-500 bg-green-500/5 border-[0.5px] border-green-500/20 px-4 py-2 rounded-lg w-fit uppercase tracking-[0.3em] shadow-[0_0_15px_rgba(34,197,94,0.05)]">
                <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.6)]" />
                API Status: Healthy
              </div>
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
