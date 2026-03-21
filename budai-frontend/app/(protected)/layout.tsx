"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { BudAIProvider, useBudAI } from "@/app/context/AppContext";
import { SidebarLeft } from "@/app/(protected)/_components/SidebarLeft";
import BudAIChat from "@/app/(protected)/_components/BudAIChat";

function SplitBrainLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const {
    accounts,
    activeAccountId,
    setActiveAccountId,
    totalBalance,
    userName,
    handleAiChartTrigger,
  } = useBudAI();
  const [revokingProviderId, setRevokingProviderId] = useState<string | null>(
    null,
  );

  const handleLogout = () => {
    localStorage.clear();
    router.push("/login");
  };

  const handleRevokeAccess = async (
    providerId: string,
    e: React.MouseEvent,
  ) => {
    e.stopPropagation();
    if (!window.confirm("Are you sure you want to disconnect this bank?"))
      return;
    setRevokingProviderId(providerId);
    const token = localStorage.getItem("budai_token") || "";
    try {
      const res = await fetch(
        `http://localhost:8080/api/accounts/${providerId}`,
        {
          method: "DELETE",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
        },
      );
      if (res.ok) window.location.reload();
    } catch (err) {
      console.log(err);
    } finally {
      setRevokingProviderId(null);
    }
  };

  const handleLinkBank = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    try {
      const token = localStorage.getItem("budai_token") || "";
      const res = await fetch(
        "http://localhost:8080/api/auth/truelayer/status",
        {
          headers: { Authorization: `Bearer ${token}` },
        },
      );
      const data = await res.json();
      if (data.auth_url) window.location.href = data.auth_url;
    } catch (err) {
      console.log(err);
    }
  };

  return (
    <div className="flex h-screen w-screen bg-[#0A120D] text-white overflow-hidden">
      <aside className="w-65 shrink-0 border-r border-[#1A2D21] bg-[#132017] z-10 h-full">
        <SidebarLeft
          accounts={accounts}
          activeAccountId={activeAccountId}
          totalBalance={totalBalance}
          userName={userName}
          revokingProviderId={revokingProviderId}
          onLogout={handleLogout}
          onSetActiveAccount={setActiveAccountId}
          onOpenLedger={() => {}}
          onRevokeAccess={handleRevokeAccess}
          onLinkBank={handleLinkBank}
        />
      </aside>
      <main className="flex-1 flex flex-col h-full relative bg-[#0A120D] overflow-hidden min-w-0">
        {children}
      </main>
      <aside className="w-100 shrink-0 border-l border-[#1A2D21] bg-[#132017] shadow-2xl z-20 h-full">
        <BudAIChat
          onAiAction={handleAiChartTrigger}
          activeAccountId={activeAccountId}
          accounts={accounts}
        />
      </aside>
    </div>
  );
}

export default function ProtectedLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <BudAIProvider>
      <SplitBrainLayout>{children}</SplitBrainLayout>
    </BudAIProvider>
  );
}
