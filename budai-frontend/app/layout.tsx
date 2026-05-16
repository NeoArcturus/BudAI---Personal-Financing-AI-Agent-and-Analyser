import "./globals.css";
import { Plus_Jakarta_Sans, JetBrains_Mono } from "next/font/google";
import { cn } from "@/lib/utils";
import { Providers } from "./providers";

const jakarta = Plus_Jakarta_Sans({
  subsets: ["latin"],
  variable: "--font-jakarta",
  weight: ["400", "500", "600", "700", "800"],
});

const jetbrains = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
});

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={cn("dark", jakarta.variable, jetbrains.variable)}>
      <body
        className="bg-[#08090D] text-white antialiased selection:bg-cyan-500/30 selection:text-cyan-200"
      >
        <Providers>
          {children}
        </Providers>
      </body>
    </html>
  );
}
