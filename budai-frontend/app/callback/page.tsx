"use client";

import { useEffect, useState, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Card, Spinner } from "@heroui/react";
import { ShieldCheck, AlertTriangle } from "lucide-react";
import { apiFetch } from "@/lib/api";

function CallbackContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [status, setStatus] = useState<"loading" | "success" | "error">("loading");
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    const code = searchParams.get("code");
    const state = searchParams.get("state");
    const error = searchParams.get("error");

    if (error) {
      setStatus("error");
      setErrorMessage("Authentication was rejected or failed.");
      setTimeout(() => router.push("/connections"), 3000);
      return;
    }

    if (!code) {
      router.push("/connections");
      return;
    }

    const processCallback = async () => {
      try {
        const res = await apiFetch("/api/auth/truelayer/callback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ code, state }),
        }, true);

        if (res.ok) {
          setStatus("success");
          setTimeout(() => {
            router.push("/connections");
          }, 2000);
        } else {
          setStatus("error");
          setErrorMessage("Failed to securely save bank tokens.");
          setTimeout(() => router.push("/connections"), 3000);
        }
      } catch (err) {
        setStatus("error");
        setErrorMessage("Network error during secure exchange.");
        setTimeout(() => router.push("/connections"), 3000);
      }
    };

    processCallback();
  }, [searchParams, router]);

  return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center p-4">
      <Card className="w-full max-w-md p-8 liquid-glass rounded-2xl flex flex-col items-center text-center gap-6 border-[0.5px] border-white/10 shadow-2xl">
        {status === "loading" && (
          <>
            <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mb-2">
              <Spinner size="lg" color="current" />
            </div>
            <div>
              <h2 className="text-xl font-black uppercase tracking-widest text-foreground">Securing Connection</h2>
              <p className="text-sm font-mono text-muted-foreground mt-2">Exchanging cryptographic tokens with TrueLayer...</p>
            </div>
          </>
        )}

        {status === "success" && (
          <>
            <div className="w-16 h-16 rounded-full bg-green-500/20 flex items-center justify-center mb-2 text-green-500 border border-green-500/30 shadow-[0_0_30px_rgba(34,197,94,0.3)]">
              <ShieldCheck size={32} />
            </div>
            <div>
              <h2 className="text-xl font-black uppercase tracking-widest text-green-500">Connection Active</h2>
              <p className="text-sm font-mono text-muted-foreground mt-2">Bank linkage securely verified. Returning to dashboard...</p>
            </div>
          </>
        )}

        {status === "error" && (
          <>
            <div className="w-16 h-16 rounded-full bg-red-500/20 flex items-center justify-center mb-2 text-red-500 border border-red-500/30 shadow-[0_0_30px_rgba(239,68,68,0.3)] animate-pulse">
              <AlertTriangle size={32} />
            </div>
            <div>
              <h2 className="text-xl font-black uppercase tracking-widest text-red-500">Exchange Failed</h2>
              <p className="text-sm font-mono text-red-500/70 mt-2">{errorMessage}</p>
              <p className="text-xs font-mono text-muted-foreground mt-4">Redirecting...</p>
            </div>
          </>
        )}
      </Card>
    </div>
  );
}

export default function CallbackPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-background flex flex-col items-center justify-center p-4">
        <Spinner size="lg" color="accent" />
      </div>
    }>
      <CallbackContent />
    </Suspense>
  );
}
