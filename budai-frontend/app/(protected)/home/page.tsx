import DashboardClient from "./DashboardClient";

export default async function HomePage() {
  // Server-Side Initial Paint Fetching
  let initialBuckets = [];
  let initialAlerts = [];
  try {
    const API_BASE_URL =
      process.env.API_BASE_URL ||
      process.env.NEXT_PUBLIC_API_BASE_URL ||
      "https://delicate-limit-4d2a.neo-arcturus07.workers.dev";
    // We try to fetch the initial data for hydration
    const [bucketsRes, alertsRes] = await Promise.all([
      fetch(`${API_BASE_URL}/api/buckets/`, { cache: "no-store" }).catch(() => null),
      fetch(`${API_BASE_URL}/api/notifications?limit=20`, { cache: "no-store" }).catch(() => null),
    ]);

    if (bucketsRes?.ok) {
      const data = await bucketsRes.json();
      initialBuckets = data.buckets || [];
    } else {
      initialBuckets = [];
    }

    if (alertsRes?.ok) {
      initialAlerts = await alertsRes.json();
    }
  } catch (err) {
    console.error("Failed to fetch initial server state", err);
  }

  return (
    <DashboardClient
      initialBuckets={initialBuckets}
      initialAlerts={initialAlerts}
    />
  );
}
