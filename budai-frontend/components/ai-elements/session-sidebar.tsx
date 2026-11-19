"use client";

import React, { useState } from "react";
import {
  Plus,
  History,
  MessageSquare,
  MoreVertical,
  Trash2,
  Edit2,
  Check,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";

export interface BudAIChatSession {
  session_id: string;
  title: string;
  last_updated: string;
}

export interface SessionSidebarProps {
  sessions: BudAIChatSession[];
  sessionsLoading: boolean;
  activeSessionId: string;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  onDeleteSession: (id: string) => void;
  onRenameSession: (id: string, newTitle: string) => Promise<void>;
  className?: string;
}

export function SessionSidebar({
  sessions,
  sessionsLoading,
  activeSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  onRenameSession,
  className,
}: SessionSidebarProps) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  const [activeMenuId, setActiveMenuId] = useState<string | null>(null);

  const startEditing = (s: BudAIChatSession) => {
    setEditingId(s.session_id);
    setEditValue(s.title);
    setActiveMenuId(null);
  };

  const handleSaveEdit = async () => {
    if (editingId && editValue.trim()) {
      await onRenameSession(editingId, editValue.trim());
      setEditingId(null);
    }
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditValue("");
  };

  return (
    <aside
      className={cn(
        "hidden md:flex md:w-64 h-full flex-col shrink-0 bg-[#0d0e14]/60 backdrop-blur-xl border-r border-white/10 transition-colors duration-300 select-none",
        className
      )}
    >
      {/* Top New Session Action */}
      <div className="p-4 border-b border-white/5">
        <button
          type="button"
          onClick={onNewSession}
          className="w-full h-10 px-4 flex items-center justify-center gap-2 rounded-xl bg-primary/10 hover:bg-primary/20 text-primary border border-primary/25 hover:border-primary/40 transition-all font-semibold text-xs shadow-[0_0_15px_rgba(0,242,255,0.08)] hover:shadow-[0_0_20px_rgba(0,242,255,0.18)] cursor-pointer group"
        >
          <Plus
            size={15}
            className="group-hover:rotate-90 transition-transform duration-300"
          />
          <span>New Session</span>
        </button>
      </div>

      {/* Header */}
      <div className="px-5 pt-4 pb-2 flex items-center gap-2 text-muted-foreground/60 text-xs font-medium">
        <History size={13} />
        <span>Recent Conversations</span>
      </div>

      {/* Session List */}
      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1 scrollbar-hide">
        {sessionsLoading ? (
          <div className="space-y-2 px-1">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="w-full h-11 rounded-xl bg-white/[0.03] animate-pulse border border-white/5"
              />
            ))}
          </div>
        ) : sessions.length === 0 ? (
          <div className="px-4 py-8 text-center text-xs text-muted-foreground/40">
            No previous sessions
          </div>
        ) : (
          sessions.map((s) => {
            const isActive = activeSessionId === s.session_id;
            const isEditing = editingId === s.session_id;

            if (isEditing) {
              return (
                <div
                  key={s.session_id}
                  className="flex items-center gap-1.5 p-1.5 rounded-xl bg-white/[0.05] border border-primary/40"
                >
                  <input
                    type="text"
                    value={editValue}
                    onChange={(e) => setEditValue(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") handleSaveEdit();
                      if (e.key === "Escape") handleCancelEdit();
                    }}
                    autoFocus
                    className="flex-1 bg-transparent text-xs text-foreground px-2 py-1 outline-none min-w-0"
                  />
                  <button
                    type="button"
                    onClick={handleSaveEdit}
                    className="p-1 rounded-md text-primary hover:bg-primary/20 transition-colors"
                    aria-label="Save title"
                  >
                    <Check size={13} />
                  </button>
                  <button
                    type="button"
                    onClick={handleCancelEdit}
                    className="p-1 rounded-md text-muted-foreground hover:bg-white/10 transition-colors"
                    aria-label="Cancel editing"
                  >
                    <X size={13} />
                  </button>
                </div>
              );
            }

            return (
              <div
                key={s.session_id}
                className={cn(
                  "group relative flex items-center justify-between rounded-xl px-3 py-2.5 text-xs transition-all cursor-pointer border",
                  isActive
                    ? "bg-primary/10 text-primary border-primary/25 shadow-xs font-medium"
                    : "text-muted-foreground hover:text-foreground hover:bg-white/[0.04] border-transparent"
                )}
                onClick={() => onSelectSession(s.session_id)}
              >
                <div className="flex items-center gap-2.5 min-w-0 flex-1">
                  <MessageSquare
                    size={14}
                    className={cn(
                      "shrink-0",
                      isActive ? "text-primary" : "text-muted-foreground/50 group-hover:text-foreground/70"
                    )}
                  />
                  <span className="truncate">
                    {s.title || "Untitled Session"}
                  </span>
                </div>

                {/* Session Action Menu Trigger */}
                <div
                  className="relative shrink-0 ml-1.5 opacity-0 group-hover:opacity-100 transition-opacity"
                  onClick={(e) => e.stopPropagation()}
                >
                  <button
                    type="button"
                    onClick={() =>
                      setActiveMenuId(activeMenuId === s.session_id ? null : s.session_id)
                    }
                    className="p-1 rounded-md hover:bg-white/10 text-muted-foreground hover:text-foreground transition-colors"
                    aria-label="Session options"
                  >
                    <MoreVertical size={13} />
                  </button>

                  {activeMenuId === s.session_id && (
                    <div
                      className="absolute right-0 top-full mt-1 w-32 rounded-xl bg-[#0d0e14] border border-white/10 p-1 shadow-2xl z-50 flex flex-col gap-0.5"
                      onMouseLeave={() => setActiveMenuId(null)}
                    >
                      <button
                        type="button"
                        onClick={() => startEditing(s)}
                        className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs text-foreground/80 hover:text-foreground hover:bg-white/10 transition-colors"
                      >
                        <Edit2 size={12} />
                        <span>Rename</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          onDeleteSession(s.session_id);
                          setActiveMenuId(null);
                        }}
                        className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs text-danger hover:bg-danger/10 transition-colors"
                      >
                        <Trash2 size={12} />
                        <span>Delete</span>
                      </button>
                    </div>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>
    </aside>
  );
}
