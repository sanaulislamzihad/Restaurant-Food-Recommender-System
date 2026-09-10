import type { Metadata } from "next";
import { Inter } from "next/font/google";

import { SiteHeader } from "@/components/site-header";
import { themeInitScript } from "@/components/theme-toggle";
import { AppProviders } from "@/providers/app-providers";

import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: "Bhoj — food picked for your taste",
    template: "%s · Bhoj",
  },
  description:
    "A restaurant ordering app with recommendations learned from what you and people like you actually order.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* Runs before paint so dark mode does not flash white on load. */}
        <script dangerouslySetInnerHTML={{ __html: themeInitScript }} />
      </head>
      <body className={`${inter.variable} font-sans`}>
        <AppProviders>
          <a
            href="#main"
            className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-accent focus:px-4 focus:py-2 focus:text-accent-foreground"
          >
            Skip to content
          </a>
          <SiteHeader />
          <main id="main" className="mx-auto max-w-6xl px-4 py-8">
            {children}
          </main>
          <footer className="border-t border-border py-8 text-center text-sm text-muted-foreground">
            <p>
              Bhoj — a demonstration recommender. Menu data is synthetic and no
              real payment is taken.
            </p>
          </footer>
        </AppProviders>
      </body>
    </html>
  );
}
