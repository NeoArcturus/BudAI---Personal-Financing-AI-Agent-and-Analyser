"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Lock, Mail } from "lucide-react";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const res = await fetch("http://localhost:8080/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const data = await res.json();
      if (res.ok && data.token) {
        localStorage.setItem("budai_token", data.token);
        router.push("/home");
      }
    } catch (err) {
      console.log(err);
    }
  };

  return (
    <div className="flex h-screen w-screen bg-[#0A120D] items-center justify-center">
      <div className="w-full max-w-md bg-[#132017] border border-[#1A2D21] p-8 rounded-3xl shadow-2xl">
        <h1 className="text-3xl font-bold text-white mb-2">Welcome Back</h1>
        <p className="text-slate-400 mb-8">Sign in to BudAI to continue.</p>
        <form onSubmit={handleLogin} className="flex flex-col gap-4">
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
            Sign In
          </button>
        </form>
        <p className="text-center text-slate-500 mt-6 text-sm">
          Don&apos;t have an account?{" "}
          <Link href="/register" className="text-[#69F0AE] hover:underline">
            Register
          </Link>
        </p>
      </div>
    </div>
  );
}
