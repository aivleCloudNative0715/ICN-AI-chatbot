import Link from 'next/link';

export default function Footer() {
  return (
    <footer className="w-full text-center p-4 bg-gray-100 border-t">
      <div className="text-sm text-gray-500">
        <span>© 2025 ICN AI Chatbot Project.</span>
        <Link href="/privacy" className="ml-4 hover:underline">
          개인정보 처리방침
        </Link>
        {/* <Link href="/terms" className="ml-4 hover:underline">
          이용약관
        </Link> */}
      </div>
    </footer>
  );
}