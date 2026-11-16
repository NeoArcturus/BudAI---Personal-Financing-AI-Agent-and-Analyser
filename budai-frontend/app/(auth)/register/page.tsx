"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { Lock, Mail, Loader2, Eye, EyeClosed, User, KeyRound, ArrowLeft } from "lucide-react";
import { authClient, verifyEmailOtp, sendVerificationOtp } from "@/lib/auth/client";
import { notifyAuthChange } from "@/lib/api";
import {
  Button,
  Card,
  InputGroup,
  InputOTP,
  Label,
  Link,
  TextField,
  Form,
  toast,
} from "@heroui/react";

export default function RegisterPage() {
  const router = useRouter();

  // Mode: "register" | "verify"
  const [mode, setMode] = useState<"register" | "verify">("register");

  // Form Fields
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [otp, setOtp] = useState("");

  const [isVisible, setIsVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isResending, setIsResending] = useState(false);

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      const { data, error } = await authClient.signUp.email({
        email,
        password,
        name,
      });

      if (error) {
        toast.danger(error.message || "Registration failed.");
        setIsLoading(false);
        return;
      }

      // If already verified or immediate session returned, continue
      if ((data as any)?.session) {
        localStorage.setItem("budai_user_name", name || email.split("@")[0] || "User");
        notifyAuthChange();
        router.push("/onboarding");
        return;
      }

      // Neon Managed Auth dispatches 6-digit code via auth@mail.myneon.app
      setMode("verify");
      toast.success("Verification code sent to your email.");
    } catch (err: any) {
      console.error("Sign up error:", err);
      toast.danger(err?.message || "Unable to complete registration. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleVerifyOtp = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (otp.length < 6) {
      toast.danger("Please enter the complete 6-digit code.");
      return;
    }

    setIsLoading(true);
    try {
      const res = await verifyEmailOtp(email, otp);

      if (res?.error) {
        toast.danger(res.error.message || "Invalid or expired security code.");
        setIsLoading(false);
        return;
      }

      localStorage.setItem("budai_user_name", name || email.split("@")[0] || "User");
      notifyAuthChange();
      toast.success("Email verified.");
      window.location.href = "/onboarding";
    } catch (err: any) {
      console.error("Verification error:", err);
      toast.danger(err?.message || "Verification failed. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleResendCode = async () => {
    setIsResending(true);
    try {
      await sendVerificationOtp(email);
      toast.success("A new security code has been sent.");
    } catch (err: any) {
      console.error("Resend error:", err);
      toast.danger(err?.message || "Unable to resend code.");
    } finally {
      setIsResending(false);
    }
  };

  return (
    <div className="flex h-screen w-full bg-transparent items-center justify-center font-sans text-white relative overflow-hidden">
      <Card className="w-full max-w-md bg-black/40 backdrop-blur-3xl border border-white/10 rounded-3xl shadow-2xl relative overflow-hidden z-10 p-8">
        <Card.Header className="flex flex-col items-start gap-1 p-0 mb-8 bg-transparent">
          <h1 className="text-3xl font-extrabold tracking-tight text-white m-0 drop-shadow-md">
            {mode === "register" ? "Create Account" : "Verify Email"}
          </h1>
          <p className="text-muted-foreground text-sm font-medium tracking-wide">
            {mode === "register"
              ? "Start managing your financial capital with verified intelligence."
              : `Enter the 6-digit verification code sent to ${email}.`}
          </p>
        </Card.Header>

        <Card.Content className="p-0 relative">
          {mode === "register" ? (
            <Form
              onSubmit={handleRegister}
              validationBehavior="native"
              className="flex flex-col gap-5 w-full z-10 relative"
            >
              <TextField className="w-full" name="Name">
                <Label className="uppercase tracking-[0.2em] text-[10px] font-black mb-2 text-primary/70 pl-1">
                  Full Name
                </Label>
                <InputGroup
                  className="bg-white/5 backdrop-blur-xl rounded-2xl flex items-center focus-within:border-primary/50 transition-all w-full border border-white/10 h-14"
                  variant="secondary"
                  isDisabled={isLoading}
                >
                  <InputGroup.Prefix className="pl-5 pr-2 text-muted-foreground flex items-center shrink-0">
                    <User size={18} />
                  </InputGroup.Prefix>
                  <InputGroup.Input
                    disabled={isLoading}
                    placeholder="Your Name"
                    type="text"
                    required
                    className="flex-1 w-full bg-transparent text-white font-medium py-3 pr-5 placeholder:text-muted-foreground/30 border-none focus:ring-0 focus:outline-none"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                  />
                </InputGroup>
              </TextField>

              <TextField className="w-full" name="Email">
                <Label className="uppercase tracking-[0.2em] text-[10px] font-black mb-2 text-primary/70 pl-1">
                  Email Address
                </Label>
                <InputGroup
                  className="bg-white/5 backdrop-blur-xl rounded-2xl flex items-center focus-within:border-primary/50 transition-all w-full border border-white/10 h-14"
                  variant="secondary"
                  isDisabled={isLoading}
                >
                  <InputGroup.Prefix className="pl-5 pr-2 text-muted-foreground flex items-center shrink-0">
                    <Mail size={18} />
                  </InputGroup.Prefix>
                  <InputGroup.Input
                    disabled={isLoading}
                    placeholder="name@email.com"
                    type="email"
                    required
                    className="flex-1 w-full bg-transparent text-white font-medium py-3 pr-5 placeholder:text-muted-foreground/30 border-none focus:ring-0 focus:outline-none"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                  />
                </InputGroup>
              </TextField>

              <TextField className="w-full" name="Password">
                <Label className="uppercase tracking-[0.2em] text-[10px] font-black mb-2 text-primary/70 pl-1">
                  Password
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
                    placeholder="••••••••"
                    type={isVisible ? "text" : "password"}
                    required
                    minLength={8}
                    className="flex-1 w-full bg-transparent text-white font-medium py-3 pr-2 placeholder:text-muted-foreground/30 border-none focus:ring-0 focus:outline-none"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
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

              <Button
                type="submit"
                isDisabled={isLoading}
                variant="secondary"
                className="w-full mt-2 font-extrabold tracking-widest rounded-2xl h-14 cursor-pointer transition-all flex items-center justify-center gap-3"
              >
                {isLoading && <Loader2 className="animate-spin" size={18} />}
                {isLoading ? "Creating Account..." : "Create Account"}
              </Button>

              <div className="mt-4 text-center">
                <p className="text-xs text-muted-foreground font-mono">
                  Already have an account?{" "}
                  <Link href="/login" className="text-primary hover:underline font-bold">
                    Sign In
                  </Link>
                </p>
              </div>
            </Form>
          ) : (
            <div className="flex flex-col gap-6 w-full z-10 relative">
              <div className="flex flex-col items-center justify-center py-4">
                <Label className="uppercase tracking-[0.2em] text-[10px] font-black mb-4 text-primary/70">
                  Security Code
                </Label>
                <InputOTP
                  maxLength={6}
                  value={otp}
                  onChange={(val: string) => setOtp(val)}
                  isDisabled={isLoading}
                >
                  <InputOTP.Group>
                    <InputOTP.Slot index={0} />
                    <InputOTP.Slot index={1} />
                    <InputOTP.Slot index={2} />
                  </InputOTP.Group>
                  <InputOTP.Separator />
                  <InputOTP.Group>
                    <InputOTP.Slot index={3} />
                    <InputOTP.Slot index={4} />
                    <InputOTP.Slot index={5} />
                  </InputOTP.Group>
                </InputOTP>
              </div>

              <Button
                onPress={() => handleVerifyOtp()}
                isDisabled={isLoading || otp.length < 6}
                variant="secondary"
                className="w-full font-extrabold tracking-widest rounded-2xl h-14 cursor-pointer transition-all flex items-center justify-center gap-3"
              >
                {isLoading && <Loader2 className="animate-spin" size={18} />}
                {isLoading ? "Verifying..." : "Confirm Code"}
              </Button>

              <div className="flex items-center justify-between text-xs text-muted-foreground pt-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onPress={() => setMode("register")}
                  className="flex items-center gap-1.5 text-muted-foreground hover:text-white p-0 border-none bg-transparent"
                >
                  <ArrowLeft size={14} />
                  Back
                </Button>

                <Button
                  variant="ghost"
                  size="sm"
                  isDisabled={isResending}
                  onPress={handleResendCode}
                  className="text-primary hover:underline font-bold p-0 border-none bg-transparent"
                >
                  {isResending ? "Sending..." : "Resend code"}
                </Button>
              </div>
            </div>
          )}
        </Card.Content>
      </Card>
    </div>
  );
}
