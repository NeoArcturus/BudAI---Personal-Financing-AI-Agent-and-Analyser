"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Lock, Mail, Loader2 } from "lucide-react";
import { apiFetch } from "@/lib/api";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    try {
      const res = await apiFetch("/api/auth/login", {
        method: "POST",
        body: JSON.stringify({ email, password }),
      });
      const data = await res.json();

      if (res.ok && data.token) {
        localStorage.setItem("budai_token", data.token);
        localStorage.setItem("budai_user_name", email.split("@")[0] || "User");
        router.push("/home");
      } else {
        setError(data.detail || "Invalid credentials.");
      }
    } catch (err) {
      console.log(err);
      setError("Unable to connect. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen w-screen bg-[#101115] items-center justify-center font-sans text-[#FFFFFF]">
      <div className="w-full max-w-md bg-[#1A1C24] border border-[#2A2D35] p-8 rounded-2xl shadow-2xl relative overflow-hidden">
        <div className="absolute top-0 right-0 w-64 h-64 bg-[#3D73FF]/5 rounded-bl-full" />

        <div className="flex items-center gap-2 mb-8 z-10 relative">
          <div className="w-8 h-8 bg-[#3D73FF] rounded-lg flex items-center justify-center">
            <span className="text-white text-sm font-bold">B</span>
          </div>
          <span className="font-bold text-xl tracking-tight">BudAI</span>
        </div>

        <h1 className="text-2xl font-bold mb-2 tracking-tight z-10 relative">
          Welcome Back
        </h1>
        <p className="text-[#8B8E98] mb-8 text-sm z-10 relative">
          Sign in to securely access your financial telemetry.
        </p>

        {error && (
          <div className="mb-6 text-[#FF5E98] text-sm font-medium bg-[#FF5E98]/10 py-3 px-4 rounded-xl border border-[#FF5E98]/20 z-10 relative">
            {error}
          </div>
        )}

        <form
          onSubmit={handleLogin}
          className="flex flex-col gap-5 z-10 relative"
        >
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-semibold text-[#8B8E98] uppercase tracking-wider">
              Email Address
            </label>
            <div className="relative">
              <Mail
                className="absolute left-4 top-3 text-[#8B8E98]"
                size={18}
              />
              <input
                type="email"
                placeholder="name@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full bg-[#101115] border border-[#2A2D35] text-white rounded-xl py-2.5 pl-11 pr-4 focus:border-[#3D73FF] focus:ring-1 focus:ring-[#3D73FF] outline-none transition-all placeholder:text-[#2A2D35]"
                required
              />
            </div>
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-semibold text-[#8B8E98] uppercase tracking-wider">
              Password
            </label>
            <div className="relative">
              <Lock
                className="absolute left-4 top-3 text-[#8B8E98]"
                size={18}
              />
              <input
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full bg-[#101115] border border-[#2A2D35] text-white rounded-xl py-2.5 pl-11 pr-4 focus:border-[#3D73FF] focus:ring-1 focus:ring-[#3D73FF] outline-none transition-all placeholder:text-[#2A2D35]"
                required
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full bg-[#3D73FF] text-white font-semibold py-3 rounded-xl mt-4 hover:bg-[#3D73FF]/90 transition-colors flex justify-center items-center gap-2 disabled:opacity-50"
          >
            {isLoading ? (
              <Loader2 size={18} className="animate-spin" />
            ) : (
              "Sign In"
            )}
          </button>
        </form>

        <p className="text-center text-[#8B8E98] mt-6 text-sm z-10 relative">
          Don&apos;t have an account?{" "}
          <Link
            href="/register"
            className="text-[#3D73FF] hover:underline font-medium"
          >
            Create one
          </Link>
        </p>
      </div>
    </div>
  );
}
