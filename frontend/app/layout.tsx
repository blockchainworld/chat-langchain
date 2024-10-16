import "./globals.css";
import type { Metadata } from "next";
import { Inter } from "next/font/google";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "StockGPT",
  description: "Chatbot for Stock Trading",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full">
      <body className={`${inter.className} h-full`}>
        <div
          className="flex flex-col h-full w-full"
          style={{ background: "rgb(10, 17, 40)" }}
        >
          {children}
        </div>
      </body>
    </html>
  );
}
