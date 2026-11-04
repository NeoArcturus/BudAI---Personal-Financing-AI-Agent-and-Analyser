import DashboardClient from "./DashboardClient";

export default async function HomePage() {
  let initialSnapshot = null;
  try {
    const API_BASE_URL =
      process.env.API_BASE_URL ||
      process.env.NEXT_PUBLIC_API_BASE_URL ||
      "http://localhost:8080";

    const res = await fetch(`${API_BASE_URL}/api/dashboard/snapshot?range_days=60`, {
      cache: "no-store",
    }).catch(() => null);

    if (res?.ok) {
      initialSnapshot = await res.json();
    }
  } catch {
    initialSnapshot = null;
  }

  return <DashboardClient initialSnapshot={initialSnapshot} />;
}
