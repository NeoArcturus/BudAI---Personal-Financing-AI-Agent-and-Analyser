"use client";

import React from "react";
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
  MessageSquare,
} from "lucide-react";
import { motion } from "framer-motion";
import Image from "next/image";

export default function LandingPage() {
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
      <nav className="fixed top-0 w-full z-50 h-24 bg-[#0c131d] border-b-[0.5px] border-white/5 flex justify-between items-center px-6 md:px-10 transition-all">
        <div className="flex items-center gap-2 shrink-0">
          <Image
            src="/FullLogo.jpg"
            alt="BudAI Logo"
            width={100}
            height={31}
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
            onPress={() => router.push("/register")}
            className="bg-linear-to-r from-[#7000ff] to-[#00f2ff] text-white font-extrabold text-xs tracking-wide px-6 h-9 rounded-md shadow-[0_0_20px_rgba(112,0,255,0.4)] hover:shadow-[0_0_30px_rgba(0,242,255,0.6)] transition-all border-none"
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
              className="text-5xl md:text-7xl font-extrabold tracking-tight text-transparent bg-clip-text bg-linear-to-b from-foreground to-foreground/70 leading-tight"
            >
              Personal Financial <br />
              <span className="text-transparent bg-clip-text bg-linear-to-r from-[#7000ff] to-[#00f2ff] drop-shadow-[0_0_25px_rgba(0,242,255,0.4)]">
                Analysis
              </span>
            </motion.h1>

            <motion.p
              variants={itemVariants}
              className="text-base md:text-lg text-foreground/50 max-w-xl mx-auto leading-relaxed font-medium tracking-tight"
            >
              Institutional-grade financial intelligence and transaction
              categorization.
            </motion.p>

            <motion.div
              variants={itemVariants}
              className="flex flex-col sm:flex-row items-center gap-6 mt-4 w-full sm:w-auto"
            >
              <Button
                onPress={() => router.push("/register")}
                className="bg-linear-to-r from-[#7000ff] to-[#00f2ff] text-white font-extrabold tracking-widest text-sm px-12 h-14 rounded-xl shadow-[0_0_30px_rgba(112,0,255,0.4)] hover:shadow-[0_0_40px_rgba(0,242,255,0.6)] hover:scale-[1.02] transition-all border-none"
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

        <section className="py-24 bg-background border-y border-white/5 relative z-20">
          <div className="text-center mb-12">
            <h3 className="text-sm font-semibold text-foreground/40 tracking-wider">
              Securely connect to your institutions
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
            <div className="text-center mb-24">
              <motion.h2
                initial={{ opacity: 0, scale: 0.95 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                className="text-4xl md:text-5xl font-extrabold text-white mb-6 tracking-tight leading-tight drop-shadow-md"
              >
                Financial Dashboard
              </motion.h2>
              <p className="text-lg text-foreground/60 max-w-2xl mx-auto font-medium tracking-wide">
                View transaction history, budget performance, and real-time
                financial projections.
              </p>
            </div>

            <div className="w-full aspect-video max-w-5xl bg-black/40 backdrop-blur-3xl rounded-[3rem] border border-white/10 p-8 relative overflow-hidden shadow-2xl flex gap-6 text-left">
              {/* Sidebar Mock */}
              <div className="w-48 h-full bg-white/5 rounded-2xl border border-white/5 p-6 flex-col gap-6 hidden md:flex">
                <div className="w-full h-8 bg-white/10 rounded-lg mb-4" />
                <div className="w-3/4 h-4 bg-white/5 rounded-md" />
                <div className="w-5/6 h-4 bg-white/5 rounded-md" />
                <div className="w-2/3 h-4 bg-white/5 rounded-md" />
                <div className="w-full h-4 bg-white/5 rounded-md mt-auto" />
              </div>

              {/* Main Content Area */}
              <div className="flex-1 h-full flex flex-col gap-6">
                {/* Top Row: Portfolio & Quick Stats */}
                <div className="flex gap-6 h-2/5">
                  {/* Mock Portfolio Card */}
                  <div className="flex-1 bg-white/3 border-[0.5px] border-primary/30 rounded-2xl p-6 flex flex-col justify-between relative overflow-hidden shadow-inner group">
                    <div className="absolute inset-0 bg-primary/10 transition-opacity" />
                    <div className="relative z-10 flex justify-between items-start">
                      <div className="flex flex-col gap-1">
                        <span className="text-[9px] font-black text-foreground/40 uppercase tracking-[0.3em]">
                          Live Account Balance
                        </span>
                        <h2 className="text-4xl font-normal tracking-tighter text-foreground mt-1 font-mono">
                          £14,250.50
                        </h2>
                      </div>
                      <div className="w-12 h-12 bg-foreground text-background rounded-xl flex items-center justify-center font-black text-xl shadow-xl">
                        B
                      </div>
                    </div>
                    <div className="relative z-10 flex justify-between items-end mt-auto">
                      <span className="text-lg font-bold tracking-tight uppercase">
                        BARCLAYS
                      </span>
                      <div className="flex items-center gap-4 text-foreground/40 text-[10px] font-bold tracking-[0.2em] font-mono">
                        <span>****1234</span>
                        <span className="opacity-20">|</span>
                        <span>20-45-14</span>
                      </div>
                    </div>
                  </div>

                  {/* Mock Spending/Cashflow */}
                  <div className="w-1/3 bg-white/5 border-[0.5px] border-white/10 rounded-2xl p-6 flex flex-col justify-between">
                    <span className="text-[9px] font-black text-foreground/40 uppercase tracking-[0.3em]">
                      Monthly Spend
                    </span>
                    <h2 className="text-2xl font-semibold tracking-tight text-foreground font-mono">
                      £2,450.00
                    </h2>
                    <div className="flex gap-2 mt-4 items-end h-16 w-full border-b border-l border-white/10 p-2">
                      {[30, 50, 40, 80, 60, 90, 45].map((h, i) => (
                        <div
                          key={i}
                          className="flex-1 bg-primary/40 rounded-t-sm border-t border-x border-primary/20"
                          style={{ height: `${h}%` }}
                        />
                      ))}
                    </div>
                  </div>
                </div>

                {/* Bottom Row: Ledger/Transactions */}
                <div className="flex-1 bg-white/5 border-[0.5px] border-white/10 rounded-2xl p-6 flex flex-col gap-4">
                  <span className="text-[9px] font-black text-foreground/40 uppercase tracking-[0.3em]">
                    Recent Transactions
                  </span>
                  <div className="flex flex-col gap-3 flex-1 overflow-hidden">
                    {[
                      {
                        name: "Tesco Extra",
                        category: "Groceries",
                        amount: "-£45.20",
                        date: "Today, 14:30",
                      },
                      {
                        name: "Transport for London",
                        category: "Travel",
                        amount: "-£12.50",
                        date: "Today, 08:45",
                      },
                      {
                        name: "Netflix",
                        category: "Entertainment",
                        amount: "-£15.99",
                        date: "Yesterday",
                      },
                    ].map((tx, i) => (
                      <div
                        key={i}
                        className="flex justify-between items-center bg-white/5 p-4 rounded-xl border border-white/5"
                      >
                        <div className="flex gap-4 items-center">
                          <div className="w-10 h-10 rounded-lg bg-white/10 flex items-center justify-center">
                            <Activity
                              size={16}
                              className="text-foreground/40"
                            />
                          </div>
                          <div className="flex flex-col">
                            <span className="text-sm font-semibold tracking-tight">
                              {tx.name}
                            </span>
                            <div className="flex items-center gap-2">
                              <span className="text-[9px] text-foreground/40 uppercase tracking-widest">
                                {tx.category}
                              </span>
                              <span className="text-[9px] text-foreground/20">
                                •
                              </span>
                              <span className="text-[9px] text-foreground/40">
                                {tx.date}
                              </span>
                            </div>
                          </div>
                        </div>
                        <span className="text-sm font-mono font-semibold tracking-tight">
                          {tx.amount}
                        </span>
                      </div>
                    ))}
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
                  title: "Intelligent Analysis",
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
                  <div className="p-3 rounded-2xl bg-primary/10 text-primary border border-primary/20">
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
                  <div className="p-3 rounded-2xl bg-primary/10 text-primary border border-primary/20">
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
                analyze spending and predict future balances using advanced
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
