"use client";

import React, { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import { toast, Modal } from "@heroui/react";
import { AlertTriangle } from "lucide-react";

interface DeleteBucketModalProps {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  bucket: any;
}

export function DeleteBucketModal({ isOpen, onOpenChange, bucket }: DeleteBucketModalProps) {
  const queryClient = useQueryClient();
  const [isDeleting, setIsDeleting] = useState(false);

  if (!bucket) return null;

  const handleDelete = async () => {
    setIsDeleting(true);
    const bucketId = bucket.id || bucket.bucket_id || bucket.bucket_uuid;
    try {
      const res = await apiFetch(`/api/buckets/${bucketId}`, { method: "DELETE" }, true);
      if (res.ok) {
        toast("Bucket deleted and funds swept to Unallocated.");
        queryClient.invalidateQueries({ queryKey: ["buckets"] });
        onOpenChange(false);
      } else {
        toast("Failed to delete bucket.", { variant: "danger" } as any);
      }
    } catch (err) {
      toast("Network error while deleting bucket.", { variant: "danger" } as any);
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <Modal.Backdrop isOpen={isOpen} onOpenChange={onOpenChange} variant="blur">
      <Modal.Container placement="center">
        <Modal.Dialog className="modal relative w-full max-w-md pointer-events-auto bg-black/90 border border-white/5 shadow-2xl rounded-2xl flex flex-col overflow-hidden">
        <div className="p-6 border-b border-white/10 flex gap-3 items-center">
          <div className="w-10 h-10 rounded-full bg-danger/20 flex items-center justify-center text-danger border border-danger/30">
            <AlertTriangle size={20} />
          </div>
          <div className="flex flex-col gap-1">
            <h3 className="text-sm font-black text-white uppercase tracking-widest">Delete Bucket</h3>
          </div>
        </div>
        <div className="p-6 py-8">
          <p className="text-foreground/70 text-sm mb-4">
            You are about to delete the <span className="text-white font-bold">{bucket.name || bucket.title}</span> bucket.
          </p>
          <div className="bg-danger/10 border border-danger/20 p-4 rounded-xl mt-2">
            <p className="text-danger text-xs font-bold uppercase tracking-wide leading-relaxed">
              Safeguard Active: Any remaining capital in this bucket will be mathematically swept back into your Unallocated balance. No funds will be lost.
            </p>
          </div>
        </div>
        <div className="p-6 border-t border-white/10 flex justify-end gap-3 bg-white/5">
          <button
            onClick={() => onOpenChange(false)}
            className="px-6 py-2 rounded-lg font-bold text-xs uppercase tracking-widest text-foreground/70 hover:bg-white/10 transition-all"
          >
            Cancel
          </button>
          <button
            onClick={handleDelete}
            disabled={isDeleting}
            className="px-6 py-2 rounded-lg font-bold text-xs uppercase tracking-widest bg-[#f31260] text-white hover:bg-[#f31260]/80 shadow-[0_0_15px_rgba(243,18,96,0.5)] transition-all disabled:opacity-50 border border-transparent"
          >
            {isDeleting ? "Deleting..." : "Confirm Delete"}
          </button>
        </div>
        </Modal.Dialog>
      </Modal.Container>
    </Modal.Backdrop>
  );
}
