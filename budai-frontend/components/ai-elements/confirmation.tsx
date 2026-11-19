"use client";

import React from "react";
import { ShieldAlert } from "lucide-react";
import { Button } from "@heroui/react";
import { cn } from "@/lib/utils";

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
  description,
  onApprove,
  onDeny,
  isPending = false,
  className,
}: ConfirmationProps) {
  return (
    <div
      className={cn(
        "w-full my-2.5 p-3.5 rounded-2xl bg-amber-500/[0.04] border border-amber-500/25 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs shadow-md backdrop-blur-md",
        className
      )}
    >
      <div className="flex items-start gap-2.5 min-w-0 flex-1">
        <div className="p-1.5 rounded-lg bg-amber-500/10 text-amber-400 border border-amber-500/20 shrink-0 mt-0.5">
          <ShieldAlert size={15} />
        </div>
        <div className="flex flex-col gap-0.5 min-w-0">
          <span className="font-bold text-foreground text-xs leading-snug">{title}</span>
          {description && (
            <span className="text-[11px] text-muted-foreground leading-relaxed">
              {description}
            </span>
          )}
        </div>
      </div>

      <div className="flex items-center gap-2 shrink-0 self-end sm:self-center">
        <Button
          size="sm"
          variant="outline"
          onPress={onDeny}
          isDisabled={isPending}
          className="text-xs font-semibold h-7 px-3 rounded-lg border-border hover:bg-foreground/5 cursor-pointer"
        >
          No
        </Button>
        <Button
          size="sm"
          onPress={onApprove}
          isPending={isPending}
          isDisabled={isPending}
          className="text-xs font-bold h-7 px-3.5 rounded-lg bg-primary text-primary-foreground hover:opacity-90 cursor-pointer shadow-xs"
        >
          Yes, proceed
        </Button>
      </div>
    </div>
  );
}
