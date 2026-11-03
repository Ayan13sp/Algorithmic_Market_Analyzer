import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-geist-sans",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Algorithmic Market Analyzer | ML-Powered Trade Classifier",
  description:
    "A quantitative analysis dashboard that trains and compares Random Forest, " +
    "Logistic Regression and SVM classifiers on engineered technical indicators " +
    "(SMA, EMA, RSI, MACD, Bollinger Bands) to generate next-day buy/sell signals " +
    "for Indian equities.",
  keywords: [
    "algorithmic trading",
    "machine learning",
    "stock analysis",
    "random forest",
    "financial engineering",
    "quantitative analysis",
  ],
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${inter.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col bg-[var(--background)]">
        {children}
      </body>
    </html>
  );
}
