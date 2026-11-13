"use client";

import React, { useState } from "react";
import { Modal, Button, TextField, Label, InputGroup, Form, toast } from "@heroui/react";
import { User, Lock, Eye, EyeClosed, Loader2, ShieldCheck } from "lucide-react";
import { authClient } from "@/lib/auth/client";

interface AccountSecurityModalProps {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  currentName: string;
  userEmail: string;
  onNameUpdated: (newName: string) => void;
}

export const AccountSecurityModal = ({
  isOpen,
  onOpenChange,
  currentName,
  userEmail,
  onNameUpdated,
}: AccountSecurityModalProps) => {
  const [activeTab, setActiveTab] = useState<"profile" | "password">("profile");

  // Profile Form State
  const [displayName, setDisplayName] = useState(currentName);
  const [isUpdatingName, setIsUpdatingName] = useState(false);

  // Password Form State
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showCurrentPassword, setShowCurrentPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [isUpdatingPassword, setIsUpdatingPassword] = useState(false);

  const handleUpdateName = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!displayName.trim()) {
      toast.danger("Name cannot be empty.");
      return;
    }

    setIsUpdatingName(true);
    try {
      if ((authClient as any).updateUser) {
        const { error } = await (authClient as any).updateUser({
          name: displayName.trim(),
        });

        if (error) {
          toast.danger(error.message || "Unable to update display name.");
          setIsUpdatingName(false);
          return;
        }
      }

      localStorage.setItem("budai_user_name", displayName.trim());
      onNameUpdated(displayName.trim());
      toast.success("Profile name updated.");
      onOpenChange(false);
    } catch (err: any) {
      console.error("Update user error:", err);
      toast.danger(err?.message || "Failed to update profile name.");
    } finally {
      setIsUpdatingName(false);
    }
  };

  const handleUpdatePassword = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!currentPassword) {
      toast.danger("Please enter your current password.");
      return;
    }

    if (newPassword.length < 8) {
      toast.danger("New password must be at least 8 characters.");
      return;
    }

    if (newPassword !== confirmPassword) {
      toast.danger("New passwords do not match.");
      return;
    }

    setIsUpdatingPassword(true);
    try {
      if ((authClient as any).changePassword) {
        const { error } = await (authClient as any).changePassword({
          currentPassword,
          newPassword,
          revokeOtherSessions: true,
        });

        if (error) {
          toast.danger(error.message || "Failed to change password.");
          setIsUpdatingPassword(false);
          return;
        }
      }

      toast.success("Password changed. Other active sessions revoked.");
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
      onOpenChange(false);
    } catch (err: any) {
      console.error("Change password error:", err);
      toast.danger(err?.message || "Failed to update password.");
    } finally {
      setIsUpdatingPassword(false);
    }
  };

  return (
    <Modal.Backdrop isOpen={isOpen} onOpenChange={onOpenChange} variant="blur">
      <Modal.Container placement="center">
        <Modal.Dialog className="modal relative w-full max-w-lg flex flex-col overflow-hidden pointer-events-auto bg-black/95 border border-white/10 shadow-2xl rounded-3xl">
          {/* Header */}
          <div className="p-6 border-b border-white/10 flex justify-between items-center bg-white/5">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center text-primary">
                <ShieldCheck size={18} />
              </div>
              <div>
                <h2 className="text-foreground font-black text-sm uppercase tracking-widest">
                  Account & Security
                </h2>
                <p className="text-xs text-muted-foreground font-medium">
                  {userEmail || "Managed Security Session"}
                </p>
              </div>
            </div>
            <button
              onClick={() => onOpenChange(false)}
              className="text-foreground/50 hover:text-white transition-colors text-lg cursor-pointer"
            >
              &times;
            </button>
          </div>

          {/* Tab Navigation */}
          <div className="flex border-b border-white/10 bg-white/[0.02] px-6 pt-2">
            <button
              onClick={() => setActiveTab("profile")}
              className={`pb-3 px-3 text-xs font-bold uppercase tracking-widest transition-all cursor-pointer border-b-2 ${
                activeTab === "profile"
                  ? "border-primary text-primary"
                  : "border-transparent text-muted-foreground hover:text-foreground"
              }`}
            >
              Profile
            </button>
            <button
              onClick={() => setActiveTab("password")}
              className={`pb-3 px-3 text-xs font-bold uppercase tracking-widest transition-all cursor-pointer border-b-2 ${
                activeTab === "password"
                  ? "border-primary text-primary"
                  : "border-transparent text-muted-foreground hover:text-foreground"
              }`}
            >
              Security & Credentials
            </button>
          </div>

          {/* Body */}
          <div className="p-6">
            {activeTab === "profile" ? (
              <Form
                onSubmit={handleUpdateName}
                validationBehavior="native"
                className="flex flex-col gap-5 w-full"
              >
                <TextField className="w-full" name="Display Name">
                  <Label className="uppercase tracking-[0.2em] text-[10px] font-black mb-2 text-primary/70 pl-1">
                    Display Name
                  </Label>
                  <InputGroup
                    className="bg-white/5 backdrop-blur-xl rounded-2xl flex items-center focus-within:border-primary/50 transition-all w-full border border-white/10 h-14"
                    variant="secondary"
                    isDisabled={isUpdatingName}
                  >
                    <InputGroup.Prefix className="pl-5 pr-2 text-muted-foreground flex items-center shrink-0">
                      <User size={18} />
                    </InputGroup.Prefix>
                    <InputGroup.Input
                      disabled={isUpdatingName}
                      placeholder="Your name"
                      type="text"
                      required
                      className="flex-1 w-full bg-transparent text-white font-medium py-3 pr-5 placeholder:text-muted-foreground/30 border-none focus:ring-0 focus:outline-none"
                      value={displayName}
                      onChange={(e) => setDisplayName(e.target.value)}
                    />
                  </InputGroup>
                </TextField>

                <div className="flex justify-end gap-3 pt-3">
                  <Button
                    variant="ghost"
                    size="sm"
                    onPress={() => onOpenChange(false)}
                    className="text-muted-foreground hover:text-white"
                  >
                    Cancel
                  </Button>
                  <Button
                    type="submit"
                    isDisabled={isUpdatingName || displayName === currentName}
                    variant="secondary"
                    className="font-bold uppercase tracking-wider text-xs px-6 py-2.5 rounded-xl cursor-pointer"
                  >
                    {isUpdatingName && <Loader2 className="animate-spin mr-1" size={14} />}
                    {isUpdatingName ? "Saving..." : "Save Name"}
                  </Button>
                </div>
              </Form>
            ) : (
              <Form
                onSubmit={handleUpdatePassword}
                validationBehavior="native"
                className="flex flex-col gap-4 w-full"
              >
                <TextField className="w-full" name="Current Password">
                  <Label className="uppercase tracking-[0.2em] text-[10px] font-black mb-1.5 text-primary/70 pl-1">
                    Current Password
                  </Label>
                  <InputGroup
                    className="bg-white/5 backdrop-blur-xl rounded-2xl flex items-center focus-within:border-primary/50 transition-all w-full border border-white/10 h-13"
                    variant="secondary"
                    isDisabled={isUpdatingPassword}
                  >
                    <InputGroup.Prefix className="pl-5 pr-2 text-muted-foreground flex items-center shrink-0">
                      <Lock size={16} />
                    </InputGroup.Prefix>
                    <InputGroup.Input
                      disabled={isUpdatingPassword}
                      placeholder="Current password"
                      type={showCurrentPassword ? "text" : "password"}
                      required
                      className="flex-1 w-full bg-transparent text-white font-medium py-2.5 pr-2 placeholder:text-muted-foreground/30 border-none focus:ring-0 focus:outline-none text-sm"
                      value={currentPassword}
                      onChange={(e) => setCurrentPassword(e.target.value)}
                    />
                    <InputGroup.Suffix className="pr-4 flex items-center shrink-0">
                      <Button
                        isIconOnly
                        type="button"
                        variant="ghost"
                        size="sm"
                        aria-label="Toggle password visibility"
                        onPress={() => setShowCurrentPassword(!showCurrentPassword)}
                        className="text-muted-foreground hover:text-foreground border-none bg-transparent"
                      >
                        {showCurrentPassword ? <Eye size={16} /> : <EyeClosed size={16} />}
                      </Button>
                    </InputGroup.Suffix>
                  </InputGroup>
                </TextField>

                <TextField className="w-full" name="New Password">
                  <Label className="uppercase tracking-[0.2em] text-[10px] font-black mb-1.5 text-primary/70 pl-1">
                    New Password
                  </Label>
                  <InputGroup
                    className="bg-white/5 backdrop-blur-xl rounded-2xl flex items-center focus-within:border-primary/50 transition-all w-full border border-white/10 h-13"
                    variant="secondary"
                    isDisabled={isUpdatingPassword}
                  >
                    <InputGroup.Prefix className="pl-5 pr-2 text-muted-foreground flex items-center shrink-0">
                      <Lock size={16} />
                    </InputGroup.Prefix>
                    <InputGroup.Input
                      disabled={isUpdatingPassword}
                      placeholder="Minimum 8 characters"
                      type={showNewPassword ? "text" : "password"}
                      required
                      minLength={8}
                      className="flex-1 w-full bg-transparent text-white font-medium py-2.5 pr-2 placeholder:text-muted-foreground/30 border-none focus:ring-0 focus:outline-none text-sm"
                      value={newPassword}
                      onChange={(e) => setNewPassword(e.target.value)}
                    />
                    <InputGroup.Suffix className="pr-4 flex items-center shrink-0">
                      <Button
                        isIconOnly
                        type="button"
                        variant="ghost"
                        size="sm"
                        aria-label="Toggle new password visibility"
                        onPress={() => setShowNewPassword(!showNewPassword)}
                        className="text-muted-foreground hover:text-foreground border-none bg-transparent"
                      >
                        {showNewPassword ? <Eye size={16} /> : <EyeClosed size={16} />}
                      </Button>
                    </InputGroup.Suffix>
                  </InputGroup>
                </TextField>

                <TextField className="w-full" name="Confirm New Password">
                  <Label className="uppercase tracking-[0.2em] text-[10px] font-black mb-1.5 text-primary/70 pl-1">
                    Confirm New Password
                  </Label>
                  <InputGroup
                    className="bg-white/5 backdrop-blur-xl rounded-2xl flex items-center focus-within:border-primary/50 transition-all w-full border border-white/10 h-13"
                    variant="secondary"
                    isDisabled={isUpdatingPassword}
                  >
                    <InputGroup.Prefix className="pl-5 pr-2 text-muted-foreground flex items-center shrink-0">
                      <Lock size={16} />
                    </InputGroup.Prefix>
                    <InputGroup.Input
                      disabled={isUpdatingPassword}
                      placeholder="Re-enter new password"
                      type={showNewPassword ? "text" : "password"}
                      required
                      minLength={8}
                      className="flex-1 w-full bg-transparent text-white font-medium py-2.5 pr-5 placeholder:text-muted-foreground/30 border-none focus:ring-0 focus:outline-none text-sm"
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                    />
                  </InputGroup>
                </TextField>

                <p className="text-[11px] text-muted-foreground/70 font-mono mt-1">
                  Changing your password will automatically terminate active sessions on other devices.
                </p>

                <div className="flex justify-end gap-3 pt-3">
                  <Button
                    variant="ghost"
                    size="sm"
                    onPress={() => onOpenChange(false)}
                    className="text-muted-foreground hover:text-white"
                  >
                    Cancel
                  </Button>
                  <Button
                    type="submit"
                    isDisabled={isUpdatingPassword || !currentPassword || newPassword.length < 8}
                    variant="secondary"
                    className="font-bold uppercase tracking-wider text-xs px-6 py-2.5 rounded-xl cursor-pointer"
                  >
                    {isUpdatingPassword && <Loader2 className="animate-spin mr-1" size={14} />}
                    {isUpdatingPassword ? "Updating..." : "Update Password"}
                  </Button>
                </div>
              </Form>
            )}
          </div>
        </Modal.Dialog>
      </Modal.Container>
    </Modal.Backdrop>
  );
};
