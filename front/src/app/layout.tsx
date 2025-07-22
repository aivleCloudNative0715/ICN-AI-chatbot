// src/app/layout.tsx
import './globals.css';
import { Inter } from 'next/font/google';
// Header, AuthModal, ChatSidebar 더 이상 직접 임포트하지 않음

const inter = Inter({ subsets: ['latin'] });

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko">
      <body className={`${inter.className} h-screen flex flex-col bg-blue-50`}>
        {children}
      </body>
    </html>
  );
}