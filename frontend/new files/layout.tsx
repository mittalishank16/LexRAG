// app/layout.tsx  —  Root Layout
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title:       "LexRAG — Indian Legal AI",
  description: "AI-powered Indian legal assistant. Ask questions about the Constitution, IPC, CrPC and more. Upload contracts for intelligent analysis.",
  icons: {
    icon: "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>⚖️</text></svg>",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ margin: 0, padding: 0, backgroundColor: "#0A0F1E" }}>
        {children}
      </body>
    </html>
  );
}
