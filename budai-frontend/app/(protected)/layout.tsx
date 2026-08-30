import { BudAIProvider, ChatSession } from "@/app/context/AppContext";
import { SearchProvider } from "@/app/context/SearchContext";
import GlobalSearchModal from "@/app/(protected)/_components/layout/SearchModal";
import GlobalChatButton from "@/app/(protected)/_components/layout/ChatButton";
import { TopNavbar } from "@/app/(protected)/_components/layout/Navbar";
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

  const baseUrl =
    process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8080";

  let initialAccounts: Account[] = [];
  let initialSessions: ChatSession[] = [];

  try {
    const [accountsRes, sessionsRes] = await Promise.all([
      fetch(`${baseUrl}/api/accounts`, {
        headers: { Authorization: `Bearer ${token}` },
        cache: 'no-store',
      }).catch((e) => {
        console.warn("Failed to fetch accounts:", e.message);
        return null;
      }),
      fetch(`${baseUrl}/api/chat/sessions`, {
        headers: { Authorization: `Bearer ${token}` },
        cache: 'no-store',
      }).catch((e) => {
        console.warn("Failed to fetch sessions:", e.message);
        return null;
      }),
    ]);

    if (accountsRes?.status !== 401 && sessionsRes?.status !== 401) {
      if (accountsRes && accountsRes.ok) {
        try {
          const accountsData = (await accountsRes.json()) as any;
          initialAccounts = accountsData.accounts || [];
        } catch (e) {
          console.warn("Failed to parse accounts JSON:", e);
        }
      }

      if (sessionsRes && sessionsRes.ok) {
        try {
          const sessionsData = (await sessionsRes.json()) as any;
          if (Array.isArray(sessionsData)) {
            initialSessions = sessionsData.map((s: any) => ({
              id: s.session_id,
              title: s.title,
              messages: [],
              lastUpdated: new Date(s.last_updated),
              contextData: s.context_data,
            }));
          }
        } catch (e) {
          console.warn("Failed to parse sessions JSON:", e);
        }
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
      <SearchProvider>
        <div className="flex flex-col h-screen w-full bg-transparent font-sans overflow-hidden relative transition-colors duration-500">

          <TopNavbar />
          <div className="relative z-10 flex-1 flex w-full h-full overflow-hidden">
            {children}
          </div>
        </div>
        <GlobalChatButton />
        <GlobalSearchModal />
      </SearchProvider>
    </BudAIProvider>
  );
}
