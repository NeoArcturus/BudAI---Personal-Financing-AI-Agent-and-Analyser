import "./globals.css";
import { Plus_Jakarta_Sans, JetBrains_Mono, Inter } from "next/font/google";
import { cn } from "@/lib/utils";
import { Providers } from "./providers";
import GlobalBackground from "@/app/(protected)/_components/GlobalBackground";

const inter = Inter({subsets:['latin'],variable:'--font-sans'});

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
    <html lang="en" className={cn("dark", jakarta.variable, jetbrains.variable, "font-sans", inter.variable)}>
      <body
        className="antialiased selection:bg-primary/30 selection:text-primary bg-black"
      >
        <GlobalBackground />
        <Providers>
          <div className="relative z-10">
            {children}
          </div>
        </Providers>
      </body>
    </html>
  );
}
