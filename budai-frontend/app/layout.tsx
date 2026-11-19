import "./globals.css";
import localFont from "next/font/local";
import { cn } from "@/lib/utils";
import { Providers } from "./providers";

const jakarta = localFont({
  src: "../public/fonts/plus-jakarta-sans.woff2",
  variable: "--font-jakarta",
  display: "swap",
});

const jetbrains = localFont({
  src: "../public/fonts/jetbrains-mono.woff2",
  variable: "--font-jetbrains",
  display: "swap",
});

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={cn(jakarta.variable, jetbrains.variable, "font-sans")} suppressHydrationWarning>
      <body
        className="antialiased selection:bg-primary/30 selection:text-primary bg-background text-foreground"
      >
        <Providers>
          <div className="relative z-10">
            {children}
          </div>
        </Providers>
      </body>
    </html>
  );
}
