import React, { useState } from "react";
import { Send, Square } from "lucide-react";
import { Button, Input } from "@heroui/react";
import { cn } from "@/lib/utils";

export const ChatInputArea = ({
  status,
  stop,
  onSend,
  isLocked,
}: {
  status: string;
  stop: () => void;
  onSend: (text: string) => void;
  isLocked?: boolean;
}) => {
  const [input, setInput] = useState("");

  const handleSend = () => {
    if (
      !input.trim() ||
      status === "streaming" ||
      status === "submitted" ||
      isLocked
    )
      return;
    onSend(input);
    setInput("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div
      className={cn(
        "relative flex items-center gap-3 border p-2 rounded-3xl bg-content1/40 backdrop-blur-xl h-auto min-h-14 px-4 transition-all",
        isLocked
          ? "border-danger/30 opacity-60 pointer-events-none bg-[repeating-linear-gradient(45deg,transparent,transparent_10px,rgba(255,255,255,0.05)_10px,rgba(255,255,255,0.05)_20px)]"
          : "border-white/10 focus-within:border-primary/50 focus-within:bg-content1/60",
      )}
    >
      <Input
        placeholder={
          isLocked
            ? "System locked awaiting parameter..."
            : "Enter financial inquiry..."
        }
        value={input}
        disabled={isLocked}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
        variant="primary"
        className="flex-1 text-sm font-medium bg-transparent shadow-none hover:bg-transparent focus-within:bg-transparent placeholder:text-foreground/30 [&>div]:bg-transparent [&>div]:shadow-none [&>div]:data-[focus=true]:bg-transparent [&>div]:data-[hover=true]:bg-transparent"
        style={{ outline: "none", boxShadow: "none" }}
      />
      {status === "streaming" || status === "submitted" ? (
        <Button
          isIconOnly
          onPress={() => stop()}
          className="bg-danger/20 text-danger w-10 h-10 min-w-10 rounded-full hover:bg-danger/30 shrink-0 cursor-pointer transition-all"
        >
          <Square size={14} fill="currentColor" />
        </Button>
      ) : (
        <Button
          isIconOnly
          onPress={handleSend}
          className="w-10 h-10 min-w-10 rounded-full shrink-0 cursor-pointer transition-all"
        >
          <Send size={16} className="ml-1" />
        </Button>
      )}
    </div>
  );
};
