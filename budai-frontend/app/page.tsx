"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";

export default function Home() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      // Updated to port 8080
      await fetch("http://localhost:8080/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });

      // Updated to port 8080
      const tlRes = await fetch("http://localhost:8080/api/truelayer/status");
      const tlData = await tlRes.json();

      if (tlData.authorized) {
        router.push("/dashboard");
      } else {
        window.location.href = tlData.auth_url;
      }
    } catch (error) {
      console.error("Authentication Error", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-[#0D1117] text-white p-6">
      <div className="bg-[#161B22] p-10 rounded-3xl border border-slate-800 w-full max-w-md shadow-2xl text-center">
        <h1 className="text-4xl font-extrabold tracking-tighter text-[#00FFAA] mb-2 animate-pulse">
          BUDAI.CORE
        </h1>
        <p className="text-sm text-slate-400 font-medium mb-8">
          Agentic Personal Finance & Behavioral Intelligence System
        </p>

        <form onSubmit={handleLogin} className="space-y-4">
          <input
            type="email"
            placeholder="BudAI ID"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            className="w-full bg-[#0D1117] border border-slate-700 rounded-xl px-4 py-3 text-white focus:border-[#00FFAA] outline-none transition-colors"
          />
          <input
            type="password"
            placeholder="Passkey"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            className="w-full bg-[#0D1117] border border-slate-700 rounded-xl px-4 py-3 text-white focus:border-[#00FFAA] outline-none transition-colors"
          />
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-[#00FFAA] text-black font-bold rounded-xl py-3 hover:scale-[1.02] transition-transform shadow-[0_0_15px_rgba(0,255,170,0.3)]"
          >
            {loading ? "Authenticating..." : "Initialize Session"}
          </button>
        </form>
      </div>
    </div>
  );
}
