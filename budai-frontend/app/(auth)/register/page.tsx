"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Lock, Mail } from "lucide-react";
import { apiFetch } from "@/lib/api";

export default function RegisterPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    try {
      const res = await apiFetch("/api/auth/register", {
        method: "POST",
        body: JSON.stringify({ email, password }),
      });
      const data = await res.json();
      if (res.ok) {
        router.push("/login");
      } else {
        setError(data.detail || "Registration failed.");
      }
    } catch (err) {
      console.log(err);
      setError("Unable to connect. Please try again.");
    }
  };

  return (
    <div className="flex h-screen w-screen bg-[#0A120D] items-center justify-center">
      <div className="w-full max-w-md bg-[#132017] border border-[#1A2D21] p-8 rounded-3xl shadow-2xl">
        <h1 className="text-3xl font-bold text-white mb-2">Create Account</h1>
        <p className="text-slate-400 mb-8">
          Join BudAI and optimize your wealth.
        </p>
        {error && (
          <div className="mb-4 text-red-500 text-sm font-bold bg-red-500/10 py-2 rounded-lg border border-red-500/50">
            {error}
          </div>
        )}
        <form onSubmit={handleRegister} className="flex flex-col gap-4">
          <div className="relative">
            <Mail
              className="absolute left-4 top-3.5 text-slate-500"
              size={18}
            />
            <input
              type="email"
              placeholder="Email Address"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full bg-[#0A120D] border border-[#1A2D21] text-white rounded-xl py-3 pl-12 pr-4 focus:border-[#69F0AE] outline-none transition-all"
              required
            />
          </div>
          <div className="relative">
            <Lock
              className="absolute left-4 top-3.5 text-slate-500"
              size={18}
            />
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full bg-[#0A120D] border border-[#1A2D21] text-white rounded-xl py-3 pl-12 pr-4 focus:border-[#69F0AE] outline-none transition-all"
              required
            />
          </div>
          <button
            type="submit"
            className="w-full bg-[#69F0AE] text-[#0A120D] font-bold py-3 rounded-xl mt-4 hover:bg-[#4ade80] transition-colors"
          >
            Register
          </button>
        </form>
        <p className="text-center text-slate-500 mt-6 text-sm">
          Already have an account?{" "}
          <Link href="/login" className="text-[#69F0AE] hover:underline">
            Sign In
          </Link>
        </p>
      </div>
    </div>
  );
}
