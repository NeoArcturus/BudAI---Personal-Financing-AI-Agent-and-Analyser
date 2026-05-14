import "./globals.css";
import { Inter, Geist } from "next/font/google";
import { cn } from "@/lib/utils";
import { Providers } from "./providers";

const geist = Geist({ subsets: ["latin"], variable: "--font-geist-sans" });
const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={cn("dark", geist.variable, inter.variable)}>
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
