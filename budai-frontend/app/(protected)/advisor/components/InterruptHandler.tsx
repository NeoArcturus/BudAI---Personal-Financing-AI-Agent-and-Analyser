import React from "react";
import { Zap } from "lucide-react";
import { Input, Button } from "@heroui/react";
import { BudAIAnnotation } from "../BudAIMessage";

export const InterruptHandler = ({
  ann,
  onRespond,
}: {
  ann: BudAIAnnotation;
  onRespond: (val: string) => void;
}) => {
  const [value, setValue] = React.useState("");

  let question = "System requires manual parameter input to proceed.";
  if (ann.type === "htil_interrupt" && ann.interrupts?.[0]?.value) {
    const val = ann.interrupts[0].value as {
      action_requests?: Array<{ args?: { question?: string } }>;
    };
    question = val?.action_requests?.[0]?.args?.question || question;
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && value.trim()) {
      onRespond(value);
    }
  };

  return (
    <div className="flex flex-col gap-4 bg-warning/10 border border-warning/40 p-6 rounded-2xl mt-4 w-full max-w-2xl relative overflow-hidden liquid-glass">
      <div className="absolute top-0 left-0 w-1 h-full bg-warning"></div>

      <div className="flex items-center gap-3 text-warning text-[10px] tracking-[0.2em] uppercase font-black">
        <Zap size={14} className="animate-pulse" />
        <span>Execution Paused • Parameter Required</span>
      </div>

      <div className="text-foreground/90 text-sm font-medium leading-relaxed">
        {question}
      </div>

      <div className="flex items-center gap-3 mt-2">
        <Input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Enter authorization or parameter..."
          className="flex-1 bg-black/40 text-sm shadow-none [&>div]:border-warning/30 [&>div]:data-[focus=true]:border-warning/60"
          variant="primary"
        />
        <Button
          className="bg-warning text-black font-bold text-[10px] tracking-widest uppercase h-10 px-6 rounded-xl hover:bg-warning/80"
          onPress={() => value.trim() && onRespond(value)}
        >
          Authorize
        </Button>
      </div>
    </div>
  );
};
