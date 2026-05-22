import { BudAIProvider, ChatSession } from "@/app/context/AppContext";
import GlobalChatButton from "@/app/(protected)/_components/GlobalChatButton";
import { Sidebar } from "@/app/(protected)/_components/Sidebar";
import { cookies } from "next/headers";
import { Account } from "@/types";

export default async function ProtectedLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const cookieStore = await cookies();
  const token = cookieStore.get("budai_token")?.value;
  const baseUrl =
    process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8080";

  let initialAccounts: Account[] = [];
  let initialSessions: ChatSession[] = [];

  if (token) {
    try {
      const accountsRes = await fetch(`${baseUrl}/api/accounts`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (accountsRes.ok) {
        const accountsData = await accountsRes.json();
        initialAccounts = accountsData.accounts || [];
      }

      const sessionsRes = await fetch(`${baseUrl}/api/chat/sessions`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (sessionsRes.ok) {
        const sessionsData = await sessionsRes.json();
        if (Array.isArray(sessionsData)) {
          initialSessions = sessionsData.map((s) => ({
            id: s.session_id,
            title: s.title,
            messages: [],
            lastUpdated: new Date(s.last_updated),
            contextData: s.context_data,
          }));
        }
      }
    } catch (e) {
      console.error("ProtectedLayout pre-fetch failed:", e);
    }
  }

  return (
    <BudAIProvider
      initialAccounts={initialAccounts}
      initialSessions={initialSessions}
    >
      <div className="flex h-screen w-full bg-transparent font-sans overflow-hidden">
        <Sidebar />
        <div className="relative z-10 flex-1 flex flex-col h-full overflow-hidden">
          <div className="flex-1 flex flex-col overflow-hidden">{children}</div>
        </div>
      </div>
      <GlobalChatButton />
    </BudAIProvider>
  );
}
