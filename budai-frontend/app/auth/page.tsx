"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { ShieldCheck, Loader2 } from "lucide-react";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState<string>("");
  const [password, setPassword] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");

  /*
   * Auth Guard: If a valid token exists in local storage, bypass login.
   */
  useEffect(() => {
    const token = localStorage.getItem("budai_token");
    if (token) router.push("/dashboard");
  }, [router]);

  /*
   * Calls the new MVC auth route. Stores the returned user_uuid as a Bearer token.
   */
  const handleLogin = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const res = await fetch("http://localhost:8080/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });

      const data = (await res.json()) as { token?: string; status?: string };

      if (data.token) {
        localStorage.setItem("budai_token", data.token);
        localStorage.setItem("budai_user_name", email.split("@")[0]);
        router.push("/dashboard");
      } else {
        setError("Invalid credentials. Please try again.");
      }
    } catch (err) {
      console.error("Login error:", err);
      setError("Failed to connect to the authentication server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0D1117] flex items-center justify-center p-4">
      <form
        onSubmit={handleLogin}
        className="bg-[#161B22] p-8 rounded-3xl border border-slate-800 w-full max-w-md shadow-2xl"
      >
        <div className="flex flex-col items-center mb-8 text-[#00FFAA]">
          <ShieldCheck size={48} className="mb-4" />
          <h1 className="text-2xl font-bold text-white tracking-widest">
            BudAI
          </h1>
          <p className="text-xs text-slate-500 uppercase">
            Secure Auth Gateway
          </p>
        </div>

        {error && (
          <div className="mb-4 text-red-500 text-sm text-center font-bold">
            {error}
          </div>
        )}

        <div className="space-y-4">
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Identity Alias (Email)"
            required
            className="w-full bg-[#0D1117] border border-slate-700 rounded-xl p-4 text-sm text-white focus:border-[#00FFAA] outline-none transition-all"
          />
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Passphrase"
            required
            className="w-full bg-[#0D1117] border border-slate-700 rounded-xl p-4 text-sm text-white focus:border-[#00FFAA] outline-none transition-all"
          />
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-[#00FFAA] text-black font-bold rounded-xl p-4 hover:bg-[#00FFAA]/80 transition-colors flex justify-center items-center"
          >
            {loading ? (
              <Loader2 className="animate-spin" size={20} />
            ) : (
              "Authenticate"
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
