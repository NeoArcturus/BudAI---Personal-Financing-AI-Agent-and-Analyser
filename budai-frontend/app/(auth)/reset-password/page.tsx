"use client";

import React, { useState, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Lock, Loader2, Eye, EyeClosed, KeyRound, ArrowLeft, CheckCircle2 } from "lucide-react";
import { authClient } from "@/lib/auth/client";
import {
  Button,
  Card,
  InputGroup,
  Label,
  Link,
  TextField,
  Form,
  toast,
} from "@heroui/react";

function ResetPasswordContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const tokenFromUrl = searchParams.get("token") || "";

  const [token, setToken] = useState(tokenFromUrl);
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [isVisible, setIsVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);

  const handleResetPassword = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!token) {
      toast.danger("Reset token is missing. Please check your recovery link.");
      return;
    }

    if (newPassword.length < 8) {
      toast.danger("Password must be at least 8 characters long.");
      return;
    }

    if (newPassword !== confirmPassword) {
      toast.danger("Passwords do not match.");
      return;
    }

    setIsLoading(true);

    try {
      if ((authClient as any).resetPassword) {
        const { error } = await (authClient as any).resetPassword({
          newPassword,
          token,
        });

        if (error) {
          toast.danger(error.message || "Failed to reset password. The link may have expired.");
          setIsLoading(false);
          return;
        }
      }

      setIsSuccess(true);
      toast.success("Password updated successfully.");
      setTimeout(() => {
        router.push("/login");
      }, 2000);
    } catch (err: any) {
      console.error("Password reset error:", err);
      toast.danger(err?.message || "Unable to reset password. Please try again.");
      setIsLoading(false);
    }
  };

  return (
    <Card className="w-full max-w-md bg-black/40 backdrop-blur-3xl border border-white/10 rounded-3xl shadow-2xl relative overflow-hidden z-10 p-8">
      <Card.Header className="flex flex-col items-start gap-1 p-0 mb-8 bg-transparent">
        <h1 className="text-3xl font-extrabold tracking-tight text-white m-0 drop-shadow-md">
          {isSuccess ? "Password Reset" : "Set New Password"}
        </h1>
        <p className="text-muted-foreground text-sm font-medium tracking-wide">
          {isSuccess
            ? "Your security credentials have been updated. Redirecting to login..."
            : "Enter your new password to regain access to your BudAI portfolio."}
        </p>
      </Card.Header>

      <Card.Content className="p-0 relative">
        {isSuccess ? (
          <div className="flex flex-col items-center justify-center py-6 gap-4">
            <div className="w-14 h-14 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center text-primary">
              <CheckCircle2 size={32} />
            </div>
            <p className="text-center text-sm text-foreground/80 font-medium">
              Your password has been changed. You can now log in with your updated credentials.
            </p>
            <Button
              variant="secondary"
              className="w-full mt-2 font-extrabold tracking-widest rounded-2xl h-14 cursor-pointer"
              onPress={() => router.push("/login")}
            >
              Go to Sign In
            </Button>
          </div>
        ) : (
          <Form
            onSubmit={handleResetPassword}
            validationBehavior="native"
            className="flex flex-col gap-5 w-full z-10 relative"
          >
            {!tokenFromUrl && (
              <TextField className="w-full" name="Token">
                <Label className="uppercase tracking-[0.2em] text-[10px] font-black mb-2 text-primary/70 pl-1">
                  Reset Token
                </Label>
                <InputGroup
                  className="bg-white/5 backdrop-blur-xl rounded-2xl flex items-center focus-within:border-primary/50 transition-all w-full border border-white/10 h-14"
                  variant="secondary"
                  isDisabled={isLoading}
                >
                  <InputGroup.Prefix className="pl-5 pr-2 text-muted-foreground flex items-center shrink-0">
                    <KeyRound size={18} />
                  </InputGroup.Prefix>
                  <InputGroup.Input
                    disabled={isLoading}
                    placeholder="Paste reset token from email"
                    type="text"
                    required
                    className="flex-1 w-full bg-transparent text-white font-medium py-3 pr-5 placeholder:text-muted-foreground/30 border-none focus:ring-0 focus:outline-none"
                    value={token}
                    onChange={(e) => setToken(e.target.value)}
                  />
                </InputGroup>
              </TextField>
            )}

            <TextField className="w-full" name="New Password">
              <Label className="uppercase tracking-[0.2em] text-[10px] font-black mb-2 text-primary/70 pl-1">
                New Password
              </Label>
              <InputGroup
                className="bg-white/5 backdrop-blur-xl rounded-2xl flex items-center focus-within:border-primary/50 transition-all w-full border border-white/10 h-14"
                variant="secondary"
                isDisabled={isLoading}
              >
                <InputGroup.Prefix className="pl-5 pr-2 text-muted-foreground flex items-center shrink-0">
                  <Lock size={18} />
                </InputGroup.Prefix>
                <InputGroup.Input
                  disabled={isLoading}
                  placeholder="Minimum 8 characters"
                  type={isVisible ? "text" : "password"}
                  required
                  minLength={8}
                  className="flex-1 w-full bg-transparent text-white font-medium py-3 pr-2 placeholder:text-muted-foreground/30 border-none focus:ring-0 focus:outline-none"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                />
                <InputGroup.Suffix className="pr-4 flex items-center shrink-0 relative z-10 pointer-events-auto">
                  <Button
                    isIconOnly
                    type="button"
                    isDisabled={isLoading}
                    aria-label={isVisible ? "Hide Password" : "Show Password"}
                    variant="ghost"
                    onPress={() => setIsVisible(!isVisible)}
                    size="sm"
                    className="hover:cursor-pointer text-muted-foreground hover:text-foreground transition-colors border-none bg-transparent"
                  >
                    {isVisible ? <Eye size={18} /> : <EyeClosed size={18} />}
                  </Button>
                </InputGroup.Suffix>
              </InputGroup>
            </TextField>

            <TextField className="w-full" name="Confirm Password">
              <Label className="uppercase tracking-[0.2em] text-[10px] font-black mb-2 text-primary/70 pl-1">
                Confirm New Password
              </Label>
              <InputGroup
                className="bg-white/5 backdrop-blur-xl rounded-2xl flex items-center focus-within:border-primary/50 transition-all w-full border border-white/10 h-14"
                variant="secondary"
                isDisabled={isLoading}
              >
                <InputGroup.Prefix className="pl-5 pr-2 text-muted-foreground flex items-center shrink-0">
                  <Lock size={18} />
                </InputGroup.Prefix>
                <InputGroup.Input
                  disabled={isLoading}
                  placeholder="Re-enter new password"
                  type={isVisible ? "text" : "password"}
                  required
                  minLength={8}
                  className="flex-1 w-full bg-transparent text-white font-medium py-3 pr-5 placeholder:text-muted-foreground/30 border-none focus:ring-0 focus:outline-none"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                />
              </InputGroup>
            </TextField>

            <Button
              type="submit"
              isDisabled={isLoading}
              variant="secondary"
              className="w-full mt-2 font-extrabold tracking-widest rounded-2xl h-14 cursor-pointer transition-all flex items-center justify-center gap-3"
            >
              {isLoading && <Loader2 className="animate-spin" size={18} />}
              {isLoading ? "Updating Password..." : "Update Password"}
            </Button>

            <div className="mt-4 text-center">
              <Link
                href="/login"
                className="text-xs text-muted-foreground hover:text-foreground font-mono inline-flex items-center gap-1.5 transition-colors"
              >
                <ArrowLeft size={14} />
                Back to Sign In
              </Link>
            </div>
          </Form>
        )}
      </Card.Content>
    </Card>
  );
}

export default function ResetPasswordPage() {
  return (
    <div className="flex h-screen w-full bg-transparent items-center justify-center font-sans text-white relative overflow-hidden">
      <Suspense
        fallback={
          <div className="flex items-center justify-center p-8">
            <Loader2 className="animate-spin text-primary" size={32} />
          </div>
        }
      >
        <ResetPasswordContent />
      </Suspense>
    </div>
  );
}
