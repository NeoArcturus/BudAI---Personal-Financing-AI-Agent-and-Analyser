"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { Lock, Mail, Loader2, Eye, EyeClosed, ArrowLeft } from "lucide-react";
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

export default function LoginPage() {
  const router = useRouter();

  // Mode: "login" | "verify" | "forgot_password"
  const [mode, setMode] = useState<"login" | "verify" | "forgot_password">("login");

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [otp, setOtp] = useState("");

  const [isVisible, setIsVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isResending, setIsResending] = useState(false);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      const { data, error } = await authClient.signIn.email({
        email,
        password,
      });

      if (error) {
        const errorMsg = (error.message || "").toLowerCase();
        if (
          errorMsg.includes("verify") ||
          errorMsg.includes("not verified") ||
          (error as any).code === "EMAIL_NOT_VERIFIED"
        ) {
          setMode("verify");
          toast.warning("Please verify your email before logging in.");
          await sendVerificationOtp(email).catch(() => {});
          setIsLoading(false);
          return;
        }

        toast.danger(error.message || "Invalid credentials.");
        setIsLoading(false);
        return;
      }

      const userName = (data as any)?.user?.name || email.split("@")[0] || "User";
      localStorage.setItem("budai_user_name", userName);
      notifyAuthChange();
      router.push("/home");
    } catch (err: any) {
      console.error("Login error:", err);
      toast.danger(err?.message || "Unable to connect. Please try again.");
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

      toast.success("Email verified successfully.");
      // Now sign in or route to home
      const { data } = await authClient.signIn.email({ email, password }).catch(() => ({ data: null }));
      const userName = (data as any)?.user?.name || email.split("@")[0] || "User";
      localStorage.setItem("budai_user_name", userName);
      notifyAuthChange();
      router.push("/home");
    } catch (err: any) {
      console.error("Verification error:", err);
      toast.danger(err?.message || "Verification failed.");
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

  const handleForgotPassword = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email) {
      toast.danger("Please enter your email address.");
      return;
    }

    setIsLoading(true);
    try {
      if ((authClient as any).forgetPassword) {
        await (authClient as any).forgetPassword({
          email,
          redirectTo: `${window.location.origin}/reset-password`,
        });
      }
      toast.success("Recovery instructions sent to your email.");
      setMode("login");
    } catch (err: any) {
      console.error("Forgot password error:", err);
      toast.danger(err?.message || "Unable to process recovery request.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen w-full bg-transparent items-center justify-center font-sans text-white relative overflow-hidden">
      <Card className="w-full max-w-md bg-black/40 backdrop-blur-3xl border border-white/10 rounded-3xl shadow-2xl relative overflow-hidden z-10 p-8">
        <Card.Header className="flex flex-col items-start gap-1 p-0 mb-8 bg-transparent">
          <h1 className="text-3xl font-extrabold tracking-tight text-white m-0 drop-shadow-md">
            {mode === "login" && "Welcome Back"}
            {mode === "verify" && "Verify Email"}
            {mode === "forgot_password" && "Reset Password"}
          </h1>
          <p className="text-muted-foreground text-sm font-medium tracking-wide">
            {mode === "login" && "Securely access your financial dashboard."}
            {mode === "verify" && `Enter the 6-digit code sent to ${email}.`}
            {mode === "forgot_password" && "Enter your email address to receive recovery instructions."}
          </p>
        </Card.Header>

        <Card.Content className="p-0 relative">
          {mode === "login" && (
            <Form
              onSubmit={handleLogin}
              validationBehavior="native"
              className="flex flex-col gap-6 w-full z-10 relative"
            >
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

              <div className="flex flex-col gap-2 w-full">
                <TextField className="w-full" name="Password">
                  <div className="flex items-center justify-between mb-2 pl-1 pr-1">
                    <Label className="uppercase tracking-[0.2em] text-[10px] font-black text-primary/70">
                      Password
                    </Label>
                    <button
                      type="button"
                      onClick={() => setMode("forgot_password")}
                      className="text-[11px] text-muted-foreground hover:text-primary transition-colors cursor-pointer"
                    >
                      Forgot password?
                    </button>
                  </div>
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
              </div>

              <Button
                type="submit"
                isDisabled={isLoading}
                variant="secondary"
                className="w-full mt-2 font-extrabold tracking-widest rounded-2xl h-14 cursor-pointer transition-all flex items-center justify-center gap-3"
              >
                {isLoading && <Loader2 className="animate-spin" size={18} />}
                {isLoading ? "Authenticating..." : "Log In"}
              </Button>

              <div className="mt-4 text-center">
                <p className="text-xs text-muted-foreground font-mono">
                  Don't have an account?{" "}
                  <Link href="/register" className="text-primary hover:underline font-bold">
                    Create Account
                  </Link>
                </p>
              </div>
            </Form>
          )}

          {mode === "verify" && (
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
                  onPress={() => setMode("login")}
                  className="flex items-center gap-1.5 text-muted-foreground hover:text-white p-0 border-none bg-transparent"
                >
                  <ArrowLeft size={14} />
                  Back to Sign In
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

          {mode === "forgot_password" && (
            <Form
              onSubmit={handleForgotPassword}
              validationBehavior="native"
              className="flex flex-col gap-6 w-full z-10 relative"
            >
              <TextField className="w-full" name="RecoveryEmail">
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

              <Button
                type="submit"
                isDisabled={isLoading}
                variant="secondary"
                className="w-full font-extrabold tracking-widest rounded-2xl h-14 cursor-pointer transition-all flex items-center justify-center gap-3"
              >
                {isLoading && <Loader2 className="animate-spin" size={18} />}
                {isLoading ? "Sending..." : "Send Recovery Link"}
              </Button>

              <div className="pt-2 text-center">
                <Button
                  variant="ghost"
                  size="sm"
                  onPress={() => setMode("login")}
                  className="flex items-center gap-1.5 mx-auto text-muted-foreground hover:text-white p-0 border-none bg-transparent"
                >
                  <ArrowLeft size={14} />
                  Back to Sign In
                </Button>
              </div>
            </Form>
          )}
        </Card.Content>
      </Card>
    </div>
  );
}
