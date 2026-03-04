"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";

export default function AuthPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();

    // Calls BudAI internal user authentication
    await fetch("http://localhost:8000/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });

    // Checks TrueLayer OAuth status to determine routing
    const tlRes = await fetch("http://localhost:8000/api/truelayer/status");
    const tlData = await tlRes.json();

    if (tlData.authorized) {
      router.push("/dashboard");
    } else {
      window.location.href = tlData.auth_url;
    }
  };

  return (
    <div className="min-h-screen bg-[#0D1117] flex items-center justify-center">
      <div className="bg-[#161B22] p-10 rounded-3xl border border-slate-800 w-96 shadow-2xl">
        <h1 className="text-3xl font-bold text-[#00FFAA] mb-8 text-center tracking-tighter">
          BUDAI.CORE
        </h1>
        <form onSubmit={handleLogin} className="space-y-6">
          <input
            type="email"
            placeholder="BudAI ID"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full bg-[#0D1117] border border-slate-700 rounded-xl px-4 py-3 text-white focus:border-[#00FFAA] outline-none"
          />
          <input
            type="password"
            placeholder="Passkey"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full bg-[#0D1117] border border-slate-700 rounded-xl px-4 py-3 text-white focus:border-[#00FFAA] outline-none"
          />
          <button
            type="submit"
            className="w-full bg-[#00FFAA] text-black font-bold rounded-xl py-3 hover:brightness-110 transition-all"
          >
            Initialize Session
          </button>
        </form>
      </div>
    </div>
  );
}
