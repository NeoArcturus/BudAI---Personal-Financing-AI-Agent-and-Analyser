"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { ShieldCheck, Loader2 } from "lucide-react";

export default function Home() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const res = await fetch("http://localhost:8080/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });

      const data = await res.json();

      // If login is successful, store token and push to dashboard immediately
      if (data.token) {
        localStorage.setItem("budai_token", data.token);
        localStorage.setItem("budai_user_name", email.split("@")[0]);
        router.push("/dashboard");
      } else {
        setError("Invalid credentials. Please try again.");
      }
    } catch (err) {
      console.error("Authentication Error", err);
      setError("Failed to connect to the authentication server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-[#0D1117] text-white p-6">
      <div className="bg-[#161B22] p-10 rounded-3xl border border-slate-800 w-full max-w-md shadow-2xl text-center">
        <div className="flex flex-col items-center mb-6 text-[#00FFAA]">
          <ShieldCheck size={48} className="mb-4" />
          <h1 className="text-4xl font-extrabold tracking-tighter mb-2 animate-pulse">
            BUDAI.CORE
          </h1>
          <p className="text-sm text-slate-400 font-medium">
            Agentic Personal Finance & Behavioral Intelligence System
          </p>
        </div>

        {error && (
          <div className="mb-4 text-red-500 text-sm font-bold bg-red-500/10 py-2 rounded-lg">
            {error}
          </div>
        )}

        <form onSubmit={handleLogin} className="space-y-4">
          <input
            type="email"
            placeholder="BudAI ID (Email)"
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
            className="w-full flex justify-center items-center gap-2 bg-[#00FFAA] text-black font-bold rounded-xl py-3 hover:scale-[1.02] transition-transform shadow-[0_0_15px_rgba(0,255,170,0.3)]"
          >
            {loading ? (
              <>
                <Loader2 className="animate-spin" size={18} /> Authenticating...
              </>
            ) : (
              "Initialize Session"
            )}
          </button>
        </form>
      </div>
    </div>
  );
}
