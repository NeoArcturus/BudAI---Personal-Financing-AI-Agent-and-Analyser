"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  Button,
  TextArea,
  CheckboxGroup,
  Checkbox,
  RadioGroup,
  Radio,
  InputGroup,
  Label,
  TextField,
  Form,
  toast,
  DatePicker,
  DateField,
  Calendar,
} from "@heroui/react";
import { DateValue, CalendarDate, today, getLocalTimeZone } from "@internationalized/date";
import { apiFetch } from "@/lib/api";
import { motion, AnimatePresence } from "framer-motion";
import { TerminalTypewriter } from "./_components/TerminalTypewriter";
import {
  ArrowRight,
  ArrowLeft,
  Target,
  Activity,
  Briefcase,
  PiggyBank,
  Banknote,
  TrendingUp,
  Building,
  CreditCard,
  GraduationCap,
  Ban,
  Database,
  Lock,
  LayoutDashboard,
  Calendar as CalendarIcon,
} from "lucide-react";

const GOAL_OPTIONS = [
  { id: "debt_paydown", label: "Debt Reduction", desc: "Aggressively pay down existing liabilities", icon: Target },
  { id: "daily_tracking", label: "Expense Tracking", desc: "Monitor daily cash flow and outflows", icon: Activity },
  { id: "freelance_management", label: "Business Management", desc: "Track variable income and business expenses", icon: Briefcase },
  { id: "saving_major", label: "Asset Accumulation", desc: "Save for significant purchases or investments", icon: PiggyBank },
];

const INCOME_OPTIONS = [
  { id: "fixed", label: "Highly Predictable", desc: "Standard salaried income with minimal variance", icon: Banknote },
  { id: "variable", label: "Variable", desc: "Freelance, gig economy, or commission-based", icon: TrendingUp },
  { id: "business", label: "Business Revenue", desc: "Corporate or small business cash flows", icon: Building },
  { id: "none", label: "No Income", desc: "Currently relying on savings, investments, or external support", icon: Ban },
];

const LIABILITY_OPTIONS = [
  { id: "student_loans", label: "Student Loans", desc: "Educational financing", icon: GraduationCap },
  { id: "credit_cards", label: "Credit Cards", desc: "Revolving high-interest debt", icon: CreditCard },
  { id: "mortgage", label: "Mortgage", desc: "Property or real estate financing", icon: Building },
];

export default function OnboardingPage() {
  const router = useRouter();

  const [currentStep, setCurrentStep] = useState(1);
  const [isTypingComplete, setIsTypingComplete] = useState(false);
  const [goals, setGoals] = useState<string[]>([]);
  const [incomePattern, setIncomePattern] = useState<string[]>([]);
  const [liabilities, setLiabilities] = useState<string[]>([]);
  const [userSummary, setUserSummary] = useState("");

  const [dobDate, setDobDate] = useState<DateValue | null>(null);
  const [employmentStatus, setEmploymentStatus] = useState("Employed");
  const [country, setCountry] = useState("United Kingdom");

  useEffect(() => {
    if (currentStep !== 7) return;

    let pollCount = 0;
    const interval = setInterval(() => {
      pollCount++;
      if (pollCount >= 3) {
        clearInterval(interval);
        router.push("/home");
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [currentStep, router]);

  const [processingState, setProcessingState] = useState<"idle" | "saving" | "generating" | "redirecting">("idle");

  const nextStep = () => {
    setIsTypingComplete(false);
    setCurrentStep((prev) => Math.min(prev + 1, 6));
  };

  const prevStep = () => {
    setIsTypingComplete(false);
    setCurrentStep((prev) => Math.max(prev - 1, 1));
  };

  const getContextHint = () => {
    const hints = [];
    if (incomePattern.includes("variable") || incomePattern.includes("business")) {
      hints.push("your average monthly income");
    }
    if (liabilities.length > 0) {
      hints.push("interest rates on your debts");
    }
    if (goals.includes("saving_major")) {
      hints.push("the amount and timeline for your savings goal");
    }

    if (hints.length > 0) {
      return `Try adding details about ${hints.join(" and ")}.`;
    }
    return "Try adding specific details about your income or expenses.";
  };

  const handleSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();

    if (!dobDate) {
      toast.danger("Please select your date of birth.");
      return;
    }

    const formattedDob = `${dobDate.year}-${String(dobDate.month).padStart(2, "0")}-${String(dobDate.day).padStart(2, "0")}`;

    setCurrentStep(6);
    setProcessingState("saving");

    try {
      const payload = {
        goals,
        income_pattern: incomePattern,
        liabilities,
        user_summary: userSummary,
        date_of_birth: formattedDob,
        employment_status: employmentStatus,
        country_of_tax_residence: country,
      };

      setProcessingState("generating");

      const onbRes = await apiFetch("/api/onboarding/complete", {
        method: "POST",
        body: JSON.stringify(payload),
      }, true);

      if (!onbRes.ok) {
        const errorData = await onbRes.json();
        toast.danger((errorData as any)?.detail || "Profile generation failed.");
        setCurrentStep(5);
        setProcessingState("idle");
        return;
      }

      setProcessingState("redirecting");

      // Fetch TrueLayer URL and redirect
      const res = await apiFetch("/api/auth/truelayer/status", {}, true);
      if (res.ok) {
        const data = await res.json() as any;
        if (data.auth_url) {
          window.location.href = data.auth_url;
          return;
        }
      }

      router.push("/home");
    } catch (error) {
      console.error("Onboarding submission error:", error);
      toast.danger("Unable to save profile. Please try again.");
      setCurrentStep(5);
      setProcessingState("idle");
    }
  };

  const deckVariants = {
    enter: { opacity: 0, y: 30, scale: 0.95 },
    center: { opacity: 1, y: 0, scale: 1 },
    exit: { opacity: 0, y: -50, scale: 1.05 }
  };

  const contentBaseClass = "group relative flex w-full flex-row items-start justify-start gap-4 rounded-xl border p-4 transition-all duration-400 ease-out cursor-pointer outline-none";
  const contentInactiveClass = "bg-white/5 border-white/10 hover:border-white/20 hover:bg-white/10 hover:scale-[1.01]";
  const contentActiveClass = "data-[selected=true]:bg-primary/10 data-[selected=true]:border-primary data-[selected=true]:shadow-[0_0_20px_rgba(0,242,255,0.15)] data-[selected=true]:scale-[1.02]";

  const iconBaseClass = "p-3 rounded-lg shrink-0 transition-colors duration-400";
  const iconInactiveClass = "bg-white/5 text-muted-foreground group-hover:text-white";
  const iconActiveClass = "group-data-[selected=true]:bg-primary group-data-[selected=true]:text-black group-data-[selected=true]:shadow-[0_0_10px_rgba(0,242,255,0.4)]";

  const labelBaseClass = "text-sm font-bold tracking-tight transition-colors duration-400 text-foreground";

  return (
    <div className="flex flex-col h-screen w-full bg-black text-foreground font-sans overflow-hidden items-center justify-center relative">
      <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-primary/10 blur-[150px] pointer-events-none" />
      <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] rounded-full bg-primary/5 blur-[150px] pointer-events-none" />

      <div className="w-full max-w-2xl px-6 relative z-10">
        <div className="backdrop-blur-3xl bg-[#0a0a0a]/60 border border-white/10 rounded-[2rem] p-10 shadow-[0_0_40px_rgba(0,0,0,0.5),inset_0_0_20px_rgba(255,255,255,0.02)] relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-b from-primary/5 to-transparent opacity-50 pointer-events-none" />

          <div className="relative z-10">
            {currentStep < 6 && (
              <div className="mb-10 flex flex-col gap-4">
                <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-[0.2em] text-primary/70 mb-1">
                  <span>Configuration Profile</span>
                  <span>Step {currentStep} of 5</span>
                </div>

                {/* Segmented Progress Indicator */}
                <div className="flex gap-2 w-full">
                  {[1, 2, 3, 4, 5].map((step) => (
                    <div
                      key={step}
                      className={`h-1.5 rounded-full flex-1 transition-all duration-500 ease-out ${step < currentStep ? "bg-primary/50" : step === currentStep ? "bg-primary shadow-[0_0_12px_rgba(0,242,255,0.6)]" : "bg-white/10"
                        }`}
                    />
                  ))}
                </div>
              </div>
            )}

            <AnimatePresence mode="wait">
              {currentStep === 1 && (
                <motion.div key="step1" variants={deckVariants} initial="enter" animate="center" exit="exit" transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }} className="flex flex-col gap-8 min-h-[300px]">
                  <div>
                    <h1 className="text-3xl font-bold tracking-tight mb-2 bg-gradient-to-br from-white to-white/40 bg-clip-text text-transparent">Primary Objectives</h1>
                    <TerminalTypewriter
                      text="Select the core financial objectives for this workspace."
                      onComplete={() => setIsTypingComplete(true)}
                      className="text-muted-foreground/70 text-sm h-6"
                    />
                  </div>

                  <div className="min-h-[250px]">
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: isTypingComplete ? 1 : 0, y: isTypingComplete ? 0 : 10, pointerEvents: isTypingComplete ? 'auto' : 'none' }}
                      transition={{ duration: 0.5 }}
                      className="flex flex-col gap-6"
                    >
                      <CheckboxGroup
                        value={goals}
                        onChange={(val) => setGoals(val as string[])}
                        className="w-full"
                      >
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full">
                          {GOAL_OPTIONS.map((opt) => (
                            <Checkbox
                              key={opt.id}
                              value={opt.id}
                              className={`${contentBaseClass} ${contentInactiveClass} ${contentActiveClass}`}
                            >
                              <div className="flex items-center gap-3 w-full">
                                <div className={`${iconBaseClass} ${iconInactiveClass} ${iconActiveClass}`}>
                                  <opt.icon size={20} />
                                </div>
                                <div className="flex flex-col gap-0.5">
                                  <span className={labelBaseClass}>{opt.label}</span>
                                  <span className="text-xs text-muted-foreground">{opt.desc}</span>
                                </div>
                              </div>
                            </Checkbox>
                          ))}
                        </div>
                      </CheckboxGroup>

                      <div className="flex justify-end mt-4">
                        <Button onPress={nextStep} isDisabled={goals.length === 0} variant="secondary" className="font-mono font-black tracking-[0.2em] text-[10px] uppercase px-12 h-14 rounded-xl flex items-center justify-center gap-2 whitespace-nowrap">
                          <span>CONTINUE</span>
                          <ArrowRight size={16} />
                        </Button>
                      </div>
                    </motion.div>
                  </div>
                </motion.div>
              )}

              {currentStep === 2 && (
                <motion.div key="step2" variants={deckVariants} initial="enter" animate="center" exit="exit" transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }} className="flex flex-col gap-8 min-h-[300px]">
                  <div>
                    <h1 className="text-3xl font-bold tracking-tight mb-2 bg-gradient-to-br from-white to-white/40 bg-clip-text text-transparent">Income Structure</h1>
                    <TerminalTypewriter
                      text="Describe your recurring cash inflow dynamics."
                      onComplete={() => setIsTypingComplete(true)}
                      className="text-muted-foreground/70 text-sm h-6"
                    />
                  </div>

                  <div className="min-h-[250px]">
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: isTypingComplete ? 1 : 0, y: isTypingComplete ? 0 : 10, pointerEvents: isTypingComplete ? 'auto' : 'none' }}
                      transition={{ duration: 0.5 }}
                      className="flex flex-col gap-6"
                    >
                      <RadioGroup
                        value={incomePattern[0] || ""}
                        onChange={(val) => setIncomePattern([val])}
                        className="w-full"
                      >
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full">
                          {INCOME_OPTIONS.map((opt) => (
                            <Radio
                              key={opt.id}
                              value={opt.id}
                              className={`${contentBaseClass} ${contentInactiveClass} ${contentActiveClass}`}
                            >
                              <div className="flex items-center gap-3 w-full">
                                <div className={`${iconBaseClass} ${iconInactiveClass} ${iconActiveClass}`}>
                                  <opt.icon size={20} />
                                </div>
                                <div className="flex flex-col gap-0.5">
                                  <span className={labelBaseClass}>{opt.label}</span>
                                  <span className="text-xs text-muted-foreground">{opt.desc}</span>
                                </div>
                              </div>
                            </Radio>
                          ))}
                        </div>
                      </RadioGroup>

                      <div className="flex justify-between mt-4">
                        <Button onPress={prevStep} variant="outline" className="border-white/10 text-foreground/60 px-12 h-14 text-[10px] font-mono font-bold uppercase tracking-[0.2em] rounded-xl bg-white/5 backdrop-blur-xl hover:bg-white/10 hover:text-foreground transition-all flex items-center justify-center gap-2 whitespace-nowrap">
                          <ArrowLeft size={16} />
                          <span>BACK</span>
                        </Button>
                        <Button onPress={nextStep} isDisabled={incomePattern.length === 0} variant="secondary" className="font-mono font-black tracking-[0.2em] text-[10px] uppercase px-12 h-14 rounded-xl flex items-center justify-center gap-2 whitespace-nowrap">
                          <span>CONTINUE</span>
                          <ArrowRight size={16} />
                        </Button>
                      </div>
                    </motion.div>
                  </div>
                </motion.div>
              )}

              {currentStep === 3 && (
                <motion.div key="step3" variants={deckVariants} initial="enter" animate="center" exit="exit" transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }} className="flex flex-col gap-8 min-h-[300px]">
                  <div>
                    <h1 className="text-3xl font-bold tracking-tight mb-2 bg-gradient-to-br from-white to-white/40 bg-clip-text text-transparent">Existing Liabilities</h1>
                    <TerminalTypewriter
                      text="Flag any structural debt or loan commitments."
                      onComplete={() => setIsTypingComplete(true)}
                      className="text-muted-foreground/70 text-sm h-6"
                    />
                  </div>

                  <div className="min-h-[250px]">
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: isTypingComplete ? 1 : 0, y: isTypingComplete ? 0 : 10, pointerEvents: isTypingComplete ? 'auto' : 'none' }}
                      transition={{ duration: 0.5 }}
                      className="flex flex-col gap-6"
                    >
                      <CheckboxGroup
                        value={liabilities}
                        onChange={(val) => setLiabilities(val as string[])}
                        className="w-full"
                      >
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 w-full">
                          {LIABILITY_OPTIONS.map((opt) => (
                            <Checkbox
                              key={opt.id}
                              value={opt.id}
                              className={`${contentBaseClass} ${contentInactiveClass} ${contentActiveClass}`}
                            >
                              <div className="flex flex-col items-center text-center gap-3 w-full p-2">
                                <div className={`${iconBaseClass} ${iconInactiveClass} ${iconActiveClass}`}>
                                  <opt.icon size={20} />
                                </div>
                                <div className="flex flex-col gap-0.5">
                                  <span className={labelBaseClass}>{opt.label}</span>
                                  <span className="text-xs text-muted-foreground">{opt.desc}</span>
                                </div>
                              </div>
                            </Checkbox>
                          ))}
                        </div>
                      </CheckboxGroup>

                      <div className="flex justify-between mt-4">
                        <Button onPress={prevStep} variant="outline" className="border-white/10 text-foreground/60 px-12 h-14 text-[10px] font-mono font-bold uppercase tracking-[0.2em] rounded-xl bg-white/5 backdrop-blur-xl hover:bg-white/10 hover:text-foreground transition-all flex items-center justify-center gap-2 whitespace-nowrap">
                          <ArrowLeft size={16} />
                          <span>BACK</span>
                        </Button>
                        <Button onPress={nextStep} variant="secondary" className="font-mono font-black tracking-[0.2em] text-[10px] uppercase px-12 h-14 rounded-xl flex items-center justify-center gap-2 whitespace-nowrap">
                          <span>CONTINUE</span>
                          <ArrowRight size={16} />
                        </Button>
                      </div>
                    </motion.div>
                  </div>
                </motion.div>
              )}

              {currentStep === 4 && (
                <motion.div key="step4" variants={deckVariants} initial="enter" animate="center" exit="exit" transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }} className="flex flex-col gap-8 min-h-[300px]">
                  <div>
                    <h1 className="text-3xl font-bold tracking-tight mb-2 bg-gradient-to-br from-white to-white/40 bg-clip-text text-transparent">Additional Details</h1>
                    <TerminalTypewriter
                      text="Tell us anything else we should know about your finances."
                      onComplete={() => setIsTypingComplete(true)}
                      className="text-muted-foreground/70 text-sm h-6"
                    />
                  </div>

                  <div className="min-h-[250px]">
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: isTypingComplete ? 1 : 0, y: isTypingComplete ? 0 : 10, pointerEvents: isTypingComplete ? 'auto' : 'none' }}
                      transition={{ duration: 0.5 }}
                      className="flex flex-col gap-4"
                    >
                      <div className="flex flex-col gap-2">
                        <TextArea
                          placeholder="e.g., I require strict monitoring of variable business expenses and aggressive paydown tracking for high-interest debt."
                          value={userSummary}
                          onChange={(e) => setUserSummary(e.target.value)}
                          rows={4}
                          className="w-full bg-white/5 border border-white/10 hover:border-white/20 focus-within:!border-primary focus-within:!bg-white/10 focus-within:shadow-[0_0_15px_rgba(0,242,255,0.2)] transition-all rounded-xl shadow-inner text-sm text-foreground placeholder:text-muted-foreground/50 p-4"
                        />
                        <p className="text-xs text-primary/80 mt-2 flex items-start gap-2 bg-primary/5 p-3 rounded-lg border border-primary/10">
                          <span className="font-bold uppercase tracking-widest shrink-0">Note:</span>
                          <span>{getContextHint()}</span>
                        </p>
                      </div>

                      <div className="flex justify-between mt-4">
                        <Button onPress={prevStep} variant="outline" className="border-white/10 text-foreground/60 px-12 h-14 text-[10px] font-mono font-bold uppercase tracking-[0.2em] rounded-xl bg-white/5 backdrop-blur-xl hover:bg-white/10 hover:text-foreground transition-all flex items-center justify-center gap-2 whitespace-nowrap">
                          <ArrowLeft size={16} />
                          <span>BACK</span>
                        </Button>
                        <Button onPress={nextStep} variant="secondary" className="font-mono font-black tracking-[0.2em] text-[10px] uppercase px-12 h-14 rounded-xl flex items-center justify-center gap-2 whitespace-nowrap">
                          <span>CONTINUE</span>
                        </Button>
                      </div>
                    </motion.div>
                  </div>
                </motion.div>
              )}

              {currentStep === 5 && (
                <motion.div key="step5" variants={deckVariants} initial="enter" animate="center" exit="exit" transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }} className="flex flex-col gap-8 min-h-[300px]">
                  <div>
                    <h1 className="text-3xl font-bold tracking-tight mb-2 bg-gradient-to-br from-white to-white/40 bg-clip-text text-transparent">Profile Details</h1>
                    <TerminalTypewriter
                      text="Provide tax residence and profile details before connecting your bank."
                      onComplete={() => setIsTypingComplete(true)}
                      className="text-muted-foreground/70 text-sm h-6"
                    />
                  </div>

                  <div className="min-h-[250px]">
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: isTypingComplete ? 1 : 0, y: isTypingComplete ? 0 : 10, pointerEvents: isTypingComplete ? 'auto' : 'none' }}
                      transition={{ duration: 0.5 }}
                      className="flex flex-col gap-4"
                    >
                      <Form validationBehavior="native" onSubmit={(e) => { e.preventDefault(); handleSubmit(); }} className="flex flex-col gap-4 w-full">
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                          <TextField className="w-full" name="Country" isRequired>
                            <Label className="uppercase tracking-[0.2em] text-[10px] font-black mb-2 text-primary/70 pl-1">Country of Tax Residence</Label>
                            <InputGroup className="bg-white/5 backdrop-blur-xl rounded-2xl flex items-center focus-within:border-primary focus-within:shadow-[0_0_15px_rgba(0,242,255,0.2)] focus-within:bg-white/10 transition-all w-full border border-white/10 h-14" variant="secondary">
                              <InputGroup.Prefix className="pl-5 pr-2 text-muted-foreground flex items-center shrink-0">
                                <Building size={18} />
                              </InputGroup.Prefix>
                              <InputGroup.Input placeholder="United Kingdom" className="flex-1 w-full bg-transparent text-white font-medium py-3 pr-5 placeholder:text-muted-foreground/30 border-none focus:ring-0 focus:outline-none" value={country} onChange={(e) => setCountry(e.target.value)} required />
                            </InputGroup>
                          </TextField>

                          <div className="w-full flex flex-col">
                            <DatePicker
                              className="w-full"
                              name="DOB"
                              isRequired
                              value={dobDate}
                              onChange={setDobDate}
                              maxValue={today(getLocalTimeZone())}
                            >
                              <Label className="uppercase tracking-[0.2em] text-[10px] font-black mb-2 text-primary/70 pl-1">
                                Date of Birth
                              </Label>
                              <DateField.Group
                                fullWidth
                                className="bg-white/5 backdrop-blur-xl rounded-2xl flex items-center focus-within:border-primary focus-within:shadow-[0_0_15px_rgba(0,242,255,0.2)] focus-within:bg-white/10 transition-all w-full border border-white/10 h-14 px-4"
                              >
                                <DateField.Prefix className="pr-3 text-muted-foreground flex items-center shrink-0">
                                  <CalendarIcon size={18} />
                                </DateField.Prefix>
                                <DateField.Input className="flex-1 text-white font-medium text-sm">
                                  {(segment) => (
                                    <DateField.Segment
                                      segment={segment}
                                      className="focus:bg-primary/20 rounded-md px-1 outline-none text-white font-mono"
                                    />
                                  )}
                                </DateField.Input>
                                <DateField.Suffix className="ml-2 flex items-center shrink-0">
                                  <DatePicker.Trigger className="text-muted-foreground hover:text-primary cursor-pointer transition-colors outline-none">
                                    <CalendarIcon size={18} />
                                  </DatePicker.Trigger>
                                </DateField.Suffix>
                              </DateField.Group>
                              <DatePicker.Popover className="popover min-w-max p-6 bg-black/95 backdrop-blur-3xl border border-white/10 rounded-3xl shadow-2xl">
                                <Calendar aria-label="Date of Birth" className="w-full min-w-[280px]">
                                  <Calendar.Header className="flex items-center gap-3 mb-6">
                                    <Calendar.YearPickerTrigger className="flex items-center gap-2 mr-auto cursor-pointer hover:opacity-70 transition-opacity outline-none">
                                      <Calendar.YearPickerTriggerHeading className="text-sm font-black uppercase tracking-widest text-primary italic" />
                                      <Calendar.YearPickerTriggerIndicator className="text-foreground/30 w-4 h-4" />
                                    </Calendar.YearPickerTrigger>
                                    <Calendar.NavButton
                                      slot="previous"
                                      className="text-primary w-8 h-8 flex items-center justify-center rounded-lg hover:bg-white/5 cursor-pointer transition-colors outline-none"
                                    />
                                    <Calendar.NavButton
                                      slot="next"
                                      className="text-primary w-8 h-8 flex items-center justify-center rounded-lg hover:bg-white/5 cursor-pointer transition-colors outline-none"
                                    />
                                  </Calendar.Header>
                                  <Calendar.Grid className="w-full border-collapse">
                                    <Calendar.GridHeader>
                                      {(day) => (
                                        <Calendar.HeaderCell className="text-[9px] font-black text-foreground/20 pb-4 text-center uppercase tracking-widest">
                                          {day}
                                        </Calendar.HeaderCell>
                                      )}
                                    </Calendar.GridHeader>
                                    <Calendar.GridBody>
                                      {(date) => (
                                        <Calendar.Cell
                                          date={date}
                                          className="w-8 h-8 flex items-center justify-center mx-auto text-xs font-mono text-foreground rounded-lg hover:bg-white/10 data-[selected=true]:bg-primary data-[selected=true]:text-primary-foreground cursor-pointer outline-none transition-all"
                                        />
                                      )}
                                    </Calendar.GridBody>
                                  </Calendar.Grid>
                                  <Calendar.YearPickerGrid>
                                    <Calendar.YearPickerGridBody>
                                      {({ year }) => (
                                        <Calendar.YearPickerCell
                                          year={year}
                                          className="h-8 px-2 w-full flex items-center justify-center mx-auto text-xs font-mono text-foreground rounded-lg hover:bg-white/10 data-[selected=true]:bg-primary data-[selected=true]:text-primary-foreground cursor-pointer outline-none transition-all"
                                        />
                                      )}
                                    </Calendar.YearPickerGridBody>
                                  </Calendar.YearPickerGrid>
                                </Calendar>
                              </DatePicker.Popover>
                            </DatePicker>
                          </div>
                        </div>

                        <TextField className="w-full" name="EmploymentStatus" isRequired>
                          <Label className="uppercase tracking-[0.2em] text-[10px] font-black mb-2 text-primary/70 pl-1">Employment Status</Label>
                          <InputGroup className="bg-white/5 backdrop-blur-xl rounded-2xl flex items-center focus-within:border-primary focus-within:shadow-[0_0_15px_rgba(0,242,255,0.2)] focus-within:bg-white/10 transition-all w-full border border-white/10 h-14" variant="secondary">
                            <InputGroup.Prefix className="pl-5 pr-2 text-muted-foreground flex items-center shrink-0">
                              <Briefcase size={18} />
                            </InputGroup.Prefix>
                            <InputGroup.Input placeholder="Employed / Self-Employed / Freelance" className="flex-1 w-full bg-transparent text-white font-medium py-3 pr-5 placeholder:text-muted-foreground/30 border-none focus:ring-0 focus:outline-none" value={employmentStatus} onChange={(e) => setEmploymentStatus(e.target.value)} required />
                          </InputGroup>
                        </TextField>

                        <div className="flex justify-between mt-4">
                          <Button onPress={prevStep} type="button" variant="outline" className="border-white/10 text-foreground/60 px-12 h-14 text-[10px] font-mono font-bold uppercase tracking-[0.2em] rounded-xl bg-white/5 backdrop-blur-xl hover:bg-white/10 hover:text-foreground transition-all flex items-center justify-center gap-2 whitespace-nowrap">
                            <ArrowLeft size={16} />
                            <span>BACK</span>
                          </Button>
                          <Button type="submit" isPending={processingState !== "idle"} variant="secondary" className="font-mono font-black tracking-[0.2em] text-[10px] uppercase px-12 h-14 rounded-xl flex items-center justify-center gap-2 whitespace-nowrap">
                            <span>CONNECT BANK ACCOUNT</span>
                          </Button>
                        </div>
                      </Form>
                    </motion.div>
                  </div>
                </motion.div>
              )}

              {currentStep === 6 && (
                <motion.div key="step6" variants={deckVariants} initial="enter" animate="center" exit="exit" transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }} className="flex flex-col gap-8 min-h-[300px] items-center justify-center text-center">
                  <div className="flex flex-col items-center gap-4">
                    <div className="relative">
                      <div className="w-16 h-16 rounded-full border-2 border-primary/20 border-t-primary animate-spin" />
                      <div className="absolute inset-0 flex items-center justify-center">
                        <Database size={24} className="text-primary animate-pulse" />
                      </div>
                    </div>
                    <div className="flex flex-col gap-1">
                      <h2 className="text-xl font-bold tracking-tight">Initializing Financial Workspace</h2>
                      <p className="text-xs text-muted-foreground font-mono">
                        {processingState === "saving" && "Recording parameters..."}
                        {processingState === "generating" && "Configuring risk corridors..."}
                        {processingState === "redirecting" && "Connecting Open Banking gateway..."}
                      </p>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  );
}
