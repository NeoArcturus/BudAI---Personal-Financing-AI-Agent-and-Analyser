import { useQuery } from "@tanstack/react-query";
import { AlertCircle, Lightbulb, Info } from "lucide-react";

interface ProactiveInsight {
  id: number;
  text: string;
  type: "warning" | "opportunity" | "info";
  urgency: number;
  created_at: string;
}

export function ProactiveInsightsFeed() {
  const { data: insights, isLoading } = useQuery({
    queryKey: ["proactive-insights"],
    queryFn: async () => {
      const res = await fetch("/api/widgets/proactive-insights");
      if (!res.ok) throw new Error("Failed to fetch insights");
      const json = await res.json();
      return json.data as ProactiveInsight[];
    },
    staleTime: 1000 * 60 * 60, // Cache for 1 hour to save network requests                 
  });

  if (isLoading || !insights?.length) return null;

  return (
    <div className="flex flex-col gap-3 mb-6">
      <h2 className="text-lg font-semibold tracking-tight">AI Briefing</h2>

      {insights.map((insight) => {
        let bgStyle = "bg-blue-500/10 border-blue-500/20 text-blue-400";
        let Icon = Info;

        if (insight.type === "warning") {
          bgStyle = "bg-red-500/10 border-red-500/20 text-red-400";
          Icon = AlertCircle;
        } else if (insight.type === "opportunity") {
          bgStyle = "bg-green-500/10 border-green-500/20 text-green-400";
          Icon = Lightbulb;
        }

        return (
          <div
            key={insight.id}
            className={`flex items-start gap-3 p-4 border rounded-xl shadow-sm ${bgStyle}`}
          >
            <Icon className="w-5 h-5 mt-0.5 shrink-0" />
            <p className="text-sm font-medium leading-relaxed">
              {insight.text}
            </p>
          </div>
        );
      })}
    </div>
  );
}
