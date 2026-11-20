"use client";

import React from "react";
import { ShieldAlert } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Item,
  ItemMedia,
  ItemContent,
  ItemTitle,
  ItemDescription,
  ItemActions,
} from "@/components/ui/item";
import { Button } from "@/components/ui/button";

export interface ConfirmationProps {
  title: string;
  description?: string;
  onApprove: () => void;
  onDeny: () => void;
  isPending?: boolean;
  className?: string;
}

export function Confirmation({
  title,
  description = "Please confirm before we make this change.",
  onApprove,
  onDeny,
  isPending = false,
  className,
}: ConfirmationProps) {
  return (
    <Item
      variant="outline"
      size="sm"
      className={cn(
        "w-full my-2.5 p-3 rounded-2xl bg-[#121d2b]/80 border-[#23354a] shadow-md backdrop-blur-md gap-3",
        className
      )}
    >
      <ItemMedia variant="icon" className="p-1.5 rounded-lg bg-amber-500/10 text-amber-400 border border-amber-500/20 shrink-0">
        <ShieldAlert size={15} />
      </ItemMedia>
      <ItemContent className="min-w-0 flex-1">
        <ItemTitle className="text-xs font-semibold text-foreground leading-snug">
          {title}
        </ItemTitle>
        {description && (
          <ItemDescription className="text-[11px] text-muted-foreground leading-relaxed mt-0.5">
            {description}
          </ItemDescription>
        )}
      </ItemContent>
      <ItemActions className="flex items-center gap-2 shrink-0">
        <Button
          size="xs"
          variant="outline"
          onClick={onDeny}
          disabled={isPending}
          className="text-xs font-medium h-7 px-3 rounded-lg border-[#2a3b52] hover:bg-white/5 cursor-pointer"
        >
          Cancel
        </Button>
        <Button
          size="xs"
          onClick={onApprove}
          disabled={isPending}
          className="text-xs font-semibold h-7 px-3.5 rounded-lg bg-primary text-primary-foreground hover:opacity-90 cursor-pointer shadow-xs"
        >
          {isPending ? "Confirming..." : "Confirm"}
        </Button>
      </ItemActions>
    </Item>
  );
}
