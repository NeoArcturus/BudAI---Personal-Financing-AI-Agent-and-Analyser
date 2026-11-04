"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { Card, Button, ProgressBar, Chip, toast } from "@heroui/react";
import { apiFetch } from "@/lib/api";
import { ShieldCheck, ArrowRight, ArrowLeft, Building2, Check } from "lucide-react";

export default function OnboardingPage() {
  const router = useRouter();

  // Current Step: 1 = Financial Baseline, 2 = Operating Guardrails, 3 = Connect Bank (Terminal)
  const [currentStep, setCurrentStep] = useState(1);

  // Step 1: Profile & Baseline
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [dob, setDob] = useState("");
  const [monthlyIncome, setMonthlyIncome] = useState("3200");
  const [paydayFrequency, setPaydayFrequency] = useState("Monthly");
  const [primaryObjective, setPrimaryObjective] = useState("daily_tracking");

  // Step 2: Guardrails & Operating Limits
  const [safetyBuffer, setSafetyBuffer] = useState("800");
  const [autonomousLimit, setAutonomousLimit] = useState("150");
  const [rebalancePolicy, setRebalancePolicy] = useState("conservative");

  // Processing state
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleNext = () => {
    if (currentStep === 1) {
      if (!name || !email || !password) {
        toast.warning("Please fill in your name, email, and password.");
        return;
      }
      setCurrentStep(2);
    } else if (currentStep === 2) {
      setCurrentStep(3);
    }
  };

  const handlePrev = () => {
    setCurrentStep((prev) => Math.max(1, prev - 1));
  };

  const handleTerminalBankConnect = async () => {
    setIsSubmitting(true);
    try {
      // Step 1: Register User
      const regRes = await apiFetch("/api/auth/register", {
        method: "POST",
        body: JSON.stringify({
          email,
          password,
          name,
          date_of_birth: dob || "1995-01-01",
          employment_status: "Employed",
          country_of_tax_residence: "GB",
        }),
      });

      if (!regRes.ok) {
        const errorData = (await regRes.json()) as { detail?: string };
        toast.danger(errorData.detail || "Registration failed.");
        setIsSubmitting(false);
        return;
      }

      // Step 2: Authenticate Session
      const loginRes = await apiFetch("/api/auth/login", {
        method: "POST",
        body: JSON.stringify({ email, password }),
      });
      const loginData = (await loginRes.json()) as { token?: string };

      if (!loginData.token) {
        toast.danger("Authentication failed after registration.");
        setIsSubmitting(false);
        return;
      }

      localStorage.setItem("budai_token", loginData.token);

      // Step 3: Save Baseline & Guardrails Configuration
      await apiFetch(
        "/api/user/profile",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            monthly_income: parseFloat(monthlyIncome),
            payday_frequency: paydayFrequency,
            primary_objective: primaryObjective,
            safety_buffer: parseFloat(safetyBuffer),
            autonomous_limit: parseFloat(autonomousLimit),
            rebalance_policy: rebalancePolicy,
          }),
        },
        true
      ).catch(() => null);

      // Step 4 (Terminal Step): Request TrueLayer OAuth Link
      const authRes = await apiFetch("/api/auth/banks/connect", { method: "POST" }, true);
      const authData = await authRes.json() as any;

      if (authData.auth_url || authData.auth_uri) {
        // Redirect directly to Open Banking authentication portal
        window.location.href = authData.auth_url || authData.auth_uri;
      } else {
        // Fallback directly to home dashboard
        router.push("/home");
      }
    } catch {
      toast.danger("Network error during initialization. Redirecting to dashboard...");
      router.push("/home");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="flex flex-col min-h-screen w-full bg-[#0b0f17] text-slate-100 font-sans items-center justify-center p-4">
      <Card className="glass-card rounded-2xl p-6 sm:p-8 border border-slate-800 shadow-2xl w-full max-w-xl space-y-6">
        {/* Header & Progress Indicator */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <span className="text-base font-semibold text-white tracking-tight">BudAI</span>
              <Chip size="sm" variant="soft" color="accent" className="font-mono text-xs">
                Setup
              </Chip>
            </div>
            <span className="text-xs font-mono text-slate-400">
              Step {currentStep} of 3
            </span>
          </div>

          <ProgressBar aria-label="Onboarding Progress" value={(currentStep / 3) * 100} className="w-full">
            <ProgressBar.Track className="bg-slate-800 h-1.5 rounded-full overflow-hidden">
              <ProgressBar.Fill className="h-full rounded-full bg-gradient-to-r from-cyan-400 to-emerald-400" />
            </ProgressBar.Track>
          </ProgressBar>
        </div>

        {/* Step 1: User Profile & Financial Baseline */}
        {currentStep === 1 && (
          <div className="space-y-4">
            <div>
              <h2 className="text-sm font-semibold text-white">Financial Baseline Setup</h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Configure your identity and baseline income rhythm.
              </p>
            </div>

            <div className="space-y-3 text-xs">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div>
                  <label className="block text-slate-400 mb-1">Full Name</label>
                  <input
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="Alex Wright"
                    className="w-full bg-slate-900 border border-slate-800 rounded-xl p-2.5 text-white"
                  />
                </div>
                <div>
                  <label className="block text-slate-400 mb-1">Date of Birth</label>
                  <input
                    type="date"
                    value={dob}
                    onChange={(e) => setDob(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-800 rounded-xl p-2.5 text-white"
                  />
                </div>
              </div>

              <div>
                <label className="block text-slate-400 mb-1">Email Address</label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="alex.wright@example.com"
                  className="w-full bg-slate-900 border border-slate-800 rounded-xl p-2.5 text-white"
                />
              </div>

              <div>
                <label className="block text-slate-400 mb-1">Password</label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••••••"
                  className="w-full bg-slate-900 border border-slate-800 rounded-xl p-2.5 text-white font-mono"
                />
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
                <div>
                  <label className="block text-slate-400 mb-1">Estimated Net Monthly Pay (£)</label>
                  <input
                    type="number"
                    value={monthlyIncome}
                    onChange={(e) => setMonthlyIncome(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-800 rounded-xl p-2.5 text-white font-mono"
                  />
                </div>
                <div>
                  <label className="block text-slate-400 mb-1">Payday Frequency</label>
                  <select
                    value={paydayFrequency}
                    onChange={(e) => setPaydayFrequency(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-800 rounded-xl p-2.5 text-white"
                  >
                    <option value="Monthly">Monthly</option>
                    <option value="Bi-weekly">Bi-weekly</option>
                    <option value="Weekly">Weekly</option>
                  </select>
                </div>
              </div>
            </div>

            <div className="flex justify-end pt-2">
              <Button
                variant="primary"
                onPress={handleNext}
                className="bg-cyan-600 hover:bg-cyan-500 text-white font-medium text-xs px-5 h-9 rounded-xl flex items-center gap-1.5"
              >
                <span>Continue</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Button>
            </div>
          </div>
        )}

        {/* Step 2: System Limits & Guardrails Configuration */}
        {currentStep === 2 && (
          <div className="space-y-4">
            <div>
              <h2 className="text-sm font-semibold text-white">Operating Limits & Guardrails</h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Set hard boundaries for autonomous transfers and cash preservation.
              </p>
            </div>

            <div className="space-y-3 text-xs">
              <Card className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 shadow-none space-y-1.5">
                <label className="block font-medium text-slate-200">
                  Minimum Buffer Floor (£)
                </label>
                <p className="text-[11px] text-slate-400">
                  Automated rebalancing is mathematically locked from dropping your operating cash below this floor.
                </p>
                <input
                  type="number"
                  value={safetyBuffer}
                  onChange={(e) => setSafetyBuffer(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-white font-mono"
                />
              </Card>

              <Card className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 shadow-none space-y-1.5">
                <label className="block font-medium text-slate-200">
                  Autonomous Transfer Limit (£)
                </label>
                <p className="text-[11px] text-slate-400">
                  Single automated adjustments above this amount require your manual confirmation.
                </p>
                <input
                  type="number"
                  value={autonomousLimit}
                  onChange={(e) => setAutonomousLimit(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-white font-mono"
                />
              </Card>

              <Card className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 shadow-none space-y-1.5">
                <label className="block font-medium text-slate-200">
                  Priority Cascade Strategy
                </label>
                <p className="text-[11px] text-slate-400">
                  When everyday cash is depleted, auxiliary pools are drained in reverse priority order.
                </p>
                <div className="flex gap-2 pt-1">
                  <Button
                    size="sm"
                    variant={rebalancePolicy === "conservative" ? "secondary" : "ghost"}
                    onPress={() => setRebalancePolicy("conservative")}
                    className={`text-xs flex-1 ${rebalancePolicy === "conservative"
                        ? "bg-slate-800 text-cyan-400 border border-slate-700"
                        : "text-slate-400"
                      }`}
                  >
                    Protect Reserves First
                  </Button>
                  <Button
                    size="sm"
                    variant={rebalancePolicy === "balanced" ? "secondary" : "ghost"}
                    onPress={() => setRebalancePolicy("balanced")}
                    className={`text-xs flex-1 ${rebalancePolicy === "balanced"
                        ? "bg-slate-800 text-cyan-400 border border-slate-700"
                        : "text-slate-400"
                      }`}
                  >
                    Balanced Cascade
                  </Button>
                </div>
              </Card>
            </div>

            <div className="flex justify-between pt-2">
              <Button
                variant="ghost"
                onPress={handlePrev}
                className="text-slate-400 hover:text-white text-xs px-4 h-9"
              >
                <ArrowLeft className="w-3.5 h-3.5 mr-1" />
                <span>Back</span>
              </Button>
              <Button
                variant="primary"
                onPress={handleNext}
                className="bg-cyan-600 hover:bg-cyan-500 text-white font-medium text-xs px-5 h-9 rounded-xl flex items-center gap-1.5"
              >
                <span>Review & Connect</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Button>
            </div>
          </div>
        )}

        {/* Step 3: Terminal Bank Connection via TrueLayer */}
        {currentStep === 3 && (
          <div className="space-y-4">
            <div>
              <h2 className="text-sm font-semibold text-white">Bank Authorization (Terminal Step)</h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Connect your physical bank via Open Banking. TrueLayer will redirect you directly to your dashboard once authorized.
              </p>
            </div>

            <div className="space-y-2 text-xs">
              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
                <div className="flex justify-between">
                  <span className="text-slate-400">Account Owner:</span>
                  <span className="text-white font-medium">{name} ({email})</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Estimated Monthly Net:</span>
                  <span className="text-white font-mono">£{parseFloat(monthlyIncome || "0").toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Protected Safety Buffer:</span>
                  <span className="text-emerald-400 font-mono">£{parseFloat(safetyBuffer || "0").toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Autonomous Action Ceiling:</span>
                  <span className="text-slate-200 font-mono">£{parseFloat(autonomousLimit || "0").toFixed(2)}</span>
                </div>
              </div>

              <div className="flex items-center gap-2 p-2.5 rounded-xl bg-emerald-950/30 border border-emerald-800/40 text-emerald-300 text-[11px]">
                <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
                <span>256-bit encrypted Open Banking 2.1 read-only protocol. BudAI cannot initiate real withdrawals.</span>
              </div>
            </div>

            <div className="flex justify-between pt-2">
              <Button
                variant="ghost"
                onPress={handlePrev}
                isDisabled={isSubmitting}
                className="text-slate-400 hover:text-white text-xs px-4 h-9"
              >
                <ArrowLeft className="w-3.5 h-3.5 mr-1" />
                <span>Back</span>
              </Button>
              <Button
                variant="primary"
                isPending={isSubmitting}
                onPress={handleTerminalBankConnect}
                className="bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-xs px-6 h-10 rounded-xl flex items-center gap-2 shadow-lg shadow-emerald-900/30"
              >
                <Building2 className="w-4 h-4" />
                <span>Connect Bank via TrueLayer</span>
              </Button>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
