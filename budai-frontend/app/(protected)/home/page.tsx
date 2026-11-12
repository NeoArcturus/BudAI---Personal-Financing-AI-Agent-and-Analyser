import { cookies } from "next/headers";
import DashboardClient from "./DashboardClient";

export default async function HomePage() {
  const cookieStore = await cookies();
  const token =
    cookieStore.get("budai_token")?.value ||
    cookieStore.get("budai_access_token")?.value ||
    cookieStore.get("better-auth.session_token")?.value ||
    cookieStore.get("access_token")?.value;

  let initialBuckets = [];
  let initialAlerts = [];
  let initialAccounts = [];
  let initialTransactions = [];
  let initialIntelligence = null;

  if (token) {
    const API_BASE_URL =
      process.env.API_BASE_URL ||
      process.env.NEXT_PUBLIC_API_BASE_URL ||
      "https://delicate-limit-4d2a.neo-arcturus07.workers.dev";
    const headers = { Authorization: `Bearer ${token}` };

    try {
      const snapshotRes = await fetch(`${API_BASE_URL}/api/dashboard/snapshot`, {
        headers,
        cache: "no-store",
      }).catch(() => null);

      if (snapshotRes?.ok) {
        const snapshot = (await snapshotRes.json()) as any;
        initialBuckets = snapshot.buckets
          ? Array.isArray(snapshot.buckets)
            ? snapshot.buckets
            : [
                snapshot.buckets.unallocated,
                ...(snapshot.buckets.bills || []),
                ...(snapshot.buckets.goals || []),
              ].filter(Boolean)
          : [];
        initialAccounts =
          snapshot.financial_anchor?.connected_accounts ||
          snapshot.accounts ||
          [];
        initialAlerts = snapshot.alerts || snapshot.notifications || [];
        initialTransactions = snapshot.recent_transactions || [];
        initialIntelligence = snapshot.intelligence || null;
      } else {
        // Fallback to parallel fetches if snapshot endpoint is unavailable
        const [bucketsRes, alertsRes, accountsRes] = await Promise.all([
          fetch(`${API_BASE_URL}/api/buckets/`, { headers, cache: "no-store" }).catch(() => null),
          fetch(`${API_BASE_URL}/api/notifications?limit=20`, { headers, cache: "no-store" }).catch(() => null),
          fetch(`${API_BASE_URL}/api/accounts`, { headers, cache: "no-store" }).catch(() => null),
        ]);

        if (bucketsRes?.ok) {
          const data = (await bucketsRes.json()) as any;
          initialBuckets = data.buckets || [];
        }

        if (alertsRes?.ok) {
          initialAlerts = (await alertsRes.json()) as any;
        }

        if (accountsRes?.ok) {
          const data = (await accountsRes.json()) as any;
          initialAccounts = data.accounts || [];
        }
      }
    } catch (err) {
      console.error("SSR Hydration fetch error:", err);
    }
  }

  return (
    <DashboardClient
      initialBuckets={initialBuckets}
      initialAlerts={initialAlerts}
      initialAccounts={initialAccounts}
      initialTransactions={initialTransactions}
      initialIntelligence={initialIntelligence}
    />
  );
}
