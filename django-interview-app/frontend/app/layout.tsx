import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Interview Emotion Assistant",
  description: "Browser camera and microphone interview assistant backed by Django"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
