import { useState } from "react";
import { useDashboardStore } from "@/store/dashboardStore";
import { SearchField } from "@heroui/react";

export function NLPDashboardController() {
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const setFilters = useDashboardStore((state) => state.setFilters);

  const handleParseIntent = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    try {
      const res = await fetch("/api/chat/parse-intent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      if (res.ok) {
        const parsedIntent = await res.json();
        // Instantly updates global state, triggering all REST widgets to refresh           
        setFilters({
          time_type: parsedIntent.time_type || "monthly",
          from_date: parsedIntent.from_date,
          to_date: parsedIntent.to_date,
          category: parsedIntent.category,
          account_ids: parsedIntent.account_ids || ["ALL"],
        });
        setQuery("");
      }
    } catch (error) {
      console.error("Failed to parse intent", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleParseIntent} className="w-full">
      <SearchField
        value={query}
        onChange={(val) => setQuery(val)}
        isReadOnly={isLoading}
      >
        <SearchField.Group className="flex flex-row border-[0.5px] rounded-xl py-2 px-4 justify-center items-center bg-white/5 border-white/10 hover:border-primary/50 transition-all shadow-inner">
          <SearchField.SearchIcon className="text-foreground/30" />
          <SearchField.Input
            placeholder={isLoading ? "Analyzing..." : "Ask anything... (e.g. 'Show my grocery spending this month')"}
            className="w-80 border-none outline-none ring-0 focus:outline-none focus:ring-0 px-3 text-[11px] font-medium tracking-wide placeholder:text-foreground/20"
          />
        </SearchField.Group>
      </SearchField>
    </form>
  );
}
