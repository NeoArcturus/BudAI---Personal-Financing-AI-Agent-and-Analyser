"use client";

import React from "react";
import { MessageSquarePlus } from "lucide-react";
import { Button } from "@heroui/react";
import { useBudAI } from "@/app/context/AppContext";
import { useRouter, usePathname } from "next/navigation";

export default function GlobalChatButton() {
  const { createNewSession } = useBudAI();
  const router = useRouter();
  const pathname = usePathname();

  if (pathname === "/advisor") return null;

  const handleOpenChat = () => {
    router.push(`/advisor`);
  };

  return (
    <div className="fixed bottom-8 right-8 z-50">
      <Button
        isIconOnly
        onPress={handleOpenChat}
        className="w-14 h-14 flex justify-center items-center rounded-full bg-linear-to-br from-primary to-primary/80 text-primary-foreground shadow-[0_10px_30px_rgba(0,242,255,0.4)] hover:scale-110 hover:shadow-[0_10px_40px_rgba(0,242,255,0.6)] transition-all cursor-pointer border-none"
      >
        <MessageSquarePlus size={24} />
      </Button>
    </div>
  );
}
