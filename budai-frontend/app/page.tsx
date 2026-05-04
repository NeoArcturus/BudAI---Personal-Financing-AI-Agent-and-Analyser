"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";
import Link from "next/link";
import { apiFetch } from "@/lib/api";

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
      const res = await apiFetch("/api/auth/login", {
        method: "POST",
        body: JSON.stringify({ email, password }),
      });

      const data = await res.json();

      if (res.ok && data.token) {
        localStorage.setItem("budai_token", data.token);
        localStorage.setItem("budai_user_name", email.split("@")[0]);
        router.push("/home");
      } else {
        setError(data.detail || "Invalid credentials. Please try again.");
      }
    } catch (err) {
      console.log(err);
      setError("Failed to connect to the authentication server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-[#101115] text-[#FFFFFF] p-6 font-sans">
      <div className="bg-[#1A1C24] p-10 rounded-2xl border border-[#2A2D35] w-full max-w-md shadow-2xl relative overflow-hidden">
        <div className="absolute top-0 right-0 w-64 h-64 bg-[#3D73FF]/5 rounded-bl-full" />

        <div className="flex flex-col items-center mb-8 z-10 relative">
          <div className="w-12 h-12 bg-[#3D73FF] rounded-xl flex items-center justify-center mb-4 shadow-[0_0_20px_rgba(61,115,255,0.3)]">
            <span className="text-white text-xl font-bold">B</span>
          </div>
          <h1 className="text-3xl font-bold tracking-tight mb-2">BudAI</h1>
          <p className="text-sm text-[#8B8E98] font-medium text-center">
            Financial Intelligence System
          </p>
        </div>

        {error && (
          <div className="mb-6 text-[#FF5E98] text-sm font-medium bg-[#FF5E98]/10 py-3 px-4 rounded-xl border border-[#FF5E98]/20 z-10 relative text-center">
            {error}
          </div>
        )}

        <form
          onSubmit={handleLogin}
          className="flex flex-col gap-5 z-10 relative"
        >
          <input
            type="email"
            placeholder="BudAI ID (Email)"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            className="w-full bg-[#101115] border border-[#2A2D35] rounded-xl px-4 py-3 text-white focus:border-[#3D73FF] focus:ring-1 focus:ring-[#3D73FF] outline-none transition-all placeholder:text-[#2A2D35]"
          />
          <input
            type="password"
            placeholder="Passkey"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            className="w-full bg-[#101115] border border-[#2A2D35] rounded-xl px-4 py-3 text-white focus:border-[#3D73FF] focus:ring-1 focus:ring-[#3D73FF] outline-none transition-all placeholder:text-[#2A2D35]"
          />
          <button
            type="submit"
            disabled={loading}
            className="w-full flex justify-center items-center gap-2 bg-[#3D73FF] text-white font-semibold rounded-xl py-3 hover:bg-[#3D73FF]/90 transition-colors disabled:opacity-50 mt-2"
          >
            {loading ? (
              <Loader2 className="animate-spin" size={18} />
            ) : (
              "Initialize Session"
            )}
          </button>
        </form>

        <p className="text-center text-[#8B8E98] mt-8 text-sm z-10 relative">
          New to BudAI?{" "}
          <Link
            href="/register"
            className="text-[#3D73FF] hover:underline font-medium"
          >
            Register here
          </Link>
        </p>
      </div>
    </div>
  );
}
