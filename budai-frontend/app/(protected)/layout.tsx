import { BudAIProvider, ChatSession } from "@/app/context/AppContext";
import GlobalChatButton from "@/app/(protected)/_components/GlobalChatButton";
import { TopNavbar } from "@/app/(protected)/_components/TopNavbar";
import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { Account } from "@/types";

export default async function ProtectedLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const cookieStore = await cookies();
  const token = cookieStore.get("budai_token")?.value;

  if (!token) {
    redirect("/login");
  }

  const baseUrl =
    process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8080";

  let initialAccounts: Account[] = [];
  let initialSessions: ChatSession[] = [];

  try {
    const [accountsRes, sessionsRes] = await Promise.all([
      fetch(`${baseUrl}/api/accounts`, {
        headers: { Authorization: `Bearer ${token}` },
      }),
      fetch(`${baseUrl}/api/chat/sessions`, {
        headers: { Authorization: `Bearer ${token}` },
      }),
    ]);

    if (accountsRes.status === 401 || sessionsRes.status === 401) {
      redirect("/login");
    }

    if (accountsRes.ok) {
      const accountsData = await accountsRes.json() as any;
      initialAccounts = accountsData.accounts || [];
    }

    if (sessionsRes.ok) {
      const sessionsData = await sessionsRes.json() as any;
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

  return (
    <BudAIProvider
      initialAccounts={initialAccounts}
      initialSessions={initialSessions}
    >
      <div className="flex flex-col h-screen w-full bg-transparent font-sans overflow-hidden relative transition-colors duration-500">
        <div className="absolute inset-0 bg-black/40 dark:bg-black/80 pointer-events-none z-0 backdrop-blur-[2px]" />
        <TopNavbar />
        <div className="relative z-10 flex-1 flex w-full h-full overflow-hidden">
          {children}
        </div>
      </div>
      <GlobalChatButton />
    </BudAIProvider>
  );
}
