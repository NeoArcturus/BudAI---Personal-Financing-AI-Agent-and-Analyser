import { createAuthClient } from "@neondatabase/auth/next";

export const authClient = createAuthClient();

export async function verifyEmailOtp(email: string, otp: string) {
  if ((authClient as any).emailOtp?.verifyEmail) {
    return await (authClient as any).emailOtp.verifyEmail({ email, otp });
  }
  return await (authClient as any).verifyEmail({ email, otp });
}

export async function sendVerificationOtp(email: string) {
  if ((authClient as any).emailOtp?.sendVerificationOtp) {
    return await (authClient as any).emailOtp.sendVerificationOtp({ email });
  }
  if ((authClient as any).sendVerificationOtp) {
    return await (authClient as any).sendVerificationOtp({ email });
  }
  return null;
}
