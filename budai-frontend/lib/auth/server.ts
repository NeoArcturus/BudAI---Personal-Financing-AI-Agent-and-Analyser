import { createNeonAuth } from "@neondatabase/auth/next/server";

export const auth = createNeonAuth({
  baseUrl:
    process.env.NEON_AUTH_BASE_URL ||
    process.env.NEXT_PUBLIC_NEON_AUTH_BASE_URL!,
  cookies: {
    secret:
      process.env.NEON_AUTH_COOKIE_SECRET ||
      "4VeB2Tqk87DOgLqDUpzN+Q7HJs0K+nbydGfGLUGtoD4=",
  },
});
