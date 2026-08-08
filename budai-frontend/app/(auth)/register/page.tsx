"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { Lock, Mail, Loader2, Eye, EyeClosed } from "lucide-react";
import { apiFetch } from "@/lib/api";
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

export default function RegisterPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isVisible, setIsVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      const res = await apiFetch("/api/auth/register", {
        method: "POST",
        body: JSON.stringify({ email, password }),
      });
      const data = (await res.json()) as { detail?: string };

      if (res.ok) {
        
        const loginRes = await apiFetch("/api/auth/login", {
          method: "POST",
          body: JSON.stringify({ email, password }),
        });
        const loginData = (await loginRes.json()) as { token?: string; detail?: string };
        
        if (loginRes.ok && loginData.token) {
          const username = email.split("@")[0] || "User";
          localStorage.removeItem(`budai_widgets_dashboard_${username}`);
          localStorage.setItem("budai_token", loginData.token);
          localStorage.setItem("budai_user_name", username);
          document.cookie = `budai_token=${loginData.token}; path=/; max-age=${60 * 60 * 24 * 7}; samesite=lax`;
          router.push("/home");
        } else {
          router.push("/login");
        }
      } else {
        toast.danger(data.detail || "Registration failed. Please try again.");
      }
    } catch (err) {
      console.error(err);
      toast.danger("Unable to connect to servers.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen w-screen bg-transparent items-center justify-center font-sans text-white relative overflow-hidden">
      <Card className="w-full max-w-md bg-black/40 backdrop-blur-3xl border border-white/10 rounded-3xl shadow-2xl relative overflow-hidden z-10 p-8">
        <Card.Header className="flex flex-col items-start gap-1 p-0 mb-8 bg-transparent">
          <h1 className="text-3xl font-extrabold tracking-tight text-white m-0 drop-shadow-md">
            Sign Up
          </h1>
          <p className="text-muted-foreground text-sm font-medium tracking-wide">
            Join BudAI for institutional-grade financial intelligence.
          </p>
        </Card.Header>

        <Card.Content className="p-0 relative">          <Form
            onSubmit={handleRegister}
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
              className="w-full mt-4 font-extrabold tracking-widest rounded-2xl h-14 cursor-pointer transition-all flex items-center justify-center gap-3 bg-primary/10 text-primary hover:bg-primary/20 border border-primary/30 shadow-lg"
            >
              {isLoading && <Loader2 className="animate-spin" size={18} />}
              {isLoading ? "Authenticating..." : "Get Started"}
            </Button>
          </Form>

          <Card.Footer className="text-center text-muted-foreground mt-10 text-[11px] z-10 relative flex justify-center w-full font-bold uppercase tracking-widest gap-2">
            <span>Already have an account?</span>
            <Link
              href="/login"
              className="text-primary hover:text-primary/80 transition-colors border-none font-black"
            >
              Log In
            </Link>
          </Card.Footer>
        </Card.Content>
      </Card>
    </div>
  );
}
