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
} from "@heroui/react";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const [isVisible, setIsVisible] = useState(false);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    try {
      const res = await apiFetch("/api/auth/login", {
        method: "POST",
        body: JSON.stringify({ email, password }),
      });
      const data = await res.json();

      if (res.ok && data.token) {
        localStorage.setItem("budai_token", data.token);
        localStorage.setItem("budai_user_name", email.split("@")[0] || "User");
        router.push("/home");
      } else {
        setError(data.detail || "Invalid credentials.");
      }
    } catch (err) {
      console.log(err);
      setError("Unable to connect. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen w-screen bg-[#101115] items-center justify-center font-sans text-white">
      <Card className="w-full max-w-md bg-[#1A1C24] border border-[#2A2D35] rounded-2xl shadow-2xl relative overflow-hidden">
        <Card.Header className="text-2xl font-bold mb-2 tracking-tight text-white bg-transparent mt-4 ml-8 flex-col items-start gap-1">
          <h1>Welcome Back</h1>
          <div className="z-10 relative">
            <p className="text-gray-500 text-sm font-normal">
              Sign in to securely access your dashboard.
            </p>
          </div>
        </Card.Header>
        <Card.Content className="p-8 pt-4 relative">
          <div className="absolute top-0 right-0 w-64 h-64 bg-[#3D73FF]/5 rounded-bl-full pointer-events-none" />

          {error && (
            <div className="mb-6 text-[#FF5E98] text-sm font-medium bg-[#FF5E98]/10 py-3 px-4 rounded-xl border border-[#FF5E98]/20 z-10 relative">
              {error}
            </div>
          )}

          <Form
            onSubmit={handleLogin}
            validationBehavior="native"
            className="flex flex-col gap-6 w-full z-10 relative"
          >
            <TextField className="w-full" name="Email">
              <Label className="uppercase tracking-wider text-xs font-semibold mb-1.5 text-cyan-400">
                Email
              </Label>
              <InputGroup
                className="bg-[#101115] rounded-2xl flex items-center focus-within:ring-1 focus-within:ring-cyan-400 w-full border border-transparent"
                variant="secondary"
              >
                <InputGroup.Prefix className="pl-4 pr-2 text-[#8B8E98] flex items-center shrink-0">
                  <Mail size={18} />
                </InputGroup.Prefix>
                <InputGroup.Input
                  placeholder="name@email.com"
                  type="email"
                  required
                  className="flex-1 w-full bg-transparent text-white py-3 pr-4 placeholder:text-[#2A2D35] border-none focus:ring-0 focus:outline-none"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
              </InputGroup>
            </TextField>

            <TextField className="w-full" name="Password">
              <Label className="uppercase tracking-wider text-xs font-semibold mb-1.5 text-cyan-400">
                Password
              </Label>
              <InputGroup
                className="bg-[#101115] rounded-2xl flex items-center focus-within:ring-1 focus-within:ring-cyan-400 w-full border border-transparent"
                variant="secondary"
              >
                <InputGroup.Prefix className="pl-4 pr-2 text-[#8B8E98] flex items-center shrink-0">
                  <Lock size={18} />
                </InputGroup.Prefix>
                <InputGroup.Input
                  placeholder="••••••••"
                  type={isVisible ? "text" : "password"}
                  required
                  className="flex-1 w-full bg-transparent text-white py-3 pr-2 placeholder:text-[#2A2D35] border-none focus:ring-0 focus:outline-none"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
                <InputGroup.Suffix className="pr-3 flex items-center shrink-0 relative z-10 pointer-events-auto">
                  <Button
                    isIconOnly
                    type="button"
                    aria-label={isVisible ? "Hide Password" : "Show Password"}
                    variant="ghost"
                    onClick={(e) => {
                      e.preventDefault();
                      setIsVisible(!isVisible);
                    }}
                    onPress={() => setIsVisible(!isVisible)}
                    size="sm"
                    className="hover:cursor-pointer text-[#8B8E98]"
                  >
                    {isVisible ? <Eye size={18} /> : <EyeClosed size={18} />}
                  </Button>
                </InputGroup.Suffix>
              </InputGroup>
            </TextField>

            <Button
              type="submit"
              className="w-full mt-2 bg-cyan-400 text-black font-semibold rounded-2xl py-3 hover:bg-cyan-300 cursor-pointer transition-colors flex items-center justify-center gap-2"
            >
              {isLoading && <Loader2 className="animate-spin" size={18} />}
              Sign In
            </Button>
          </Form>

          <Card.Footer className="text-center text-gray-400 mt-8 text-sm z-10 relative flex justify-center w-full">
            <span>
              Don&apos;t have an account?{" "}
              <Link
                href="/register"
                className="inline-flex items-center gap-1 text-cyan-400 hover:underline font-bold"
              >
                Create one
                <Link.Icon />
              </Link>
            </span>
          </Card.Footer>
        </Card.Content>
      </Card>
    </div>
  );
}
