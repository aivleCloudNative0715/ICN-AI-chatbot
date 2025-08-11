// src/components/chat/ChatBubble.tsx
import React from 'react';
import { Button } from 'primereact/button';
import { PencilIcon, DocumentDuplicateIcon, ArrowPathIcon } from '@heroicons/react/24/outline';

// ✨ ChatBubble이 받을 props 타입을 확장
interface ChatBubbleProps {
  // message 텍스트 대신 전체 메시지 객체를 받도록 변경
  message: {
    messageId: string;
    content: string;
    sender: 'user' | 'chatbot';
    userMessageId: string | null; // 챗봇 답변의 경우, 원본 질문 ID
  };
  // 수정과 재생성 핸들러를 props로 받음
  onEdit: (originalMessageId: string, currentContent: string) => void;
  onRegenerate: (originalUserMessageId: string) => void;
}

export default function ChatBubble({ message, onEdit, onRegenerate }: ChatBubbleProps) {
  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
  };

  const handleEditClick = () => {
    const newContent = prompt('질문을 수정해주세요:', message.content);
    if (newContent && newContent.trim() !== '') {
      onEdit(message.messageId, newContent);
    }
  };
  
  const handleRegenerateClick = () => {
    if (message.userMessageId) {
       onRegenerate(message.userMessageId);
    }
  };

  return (
    <div className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'} mb-4`}>
      <div
        className={`flex flex-col max-w-lg p-3 rounded-xl ${
          message.sender === 'user'
            ? 'bg-blue-200 text-gray-800 rounded-br-none items-end shadow-md'
            : 'bg-blue-50 text-gray-800 rounded-bl-none'
        }`}
      >
        <p className="text-sm sm:text-base mb-2">{message.content}</p>

        {/* 사용자 메시지일 경우: 수정, 복사 버튼 */}
        {message.sender === 'user' && (
          <div className="flex gap-3 text-xs text-gray-500 mt-2">
            <Button
              icon={<PencilIcon className="h-4 w-4" />}
              tooltip="편집"
              tooltipOptions={{ position: 'top' }}
              className="!text-gray-500 w-fit"
              onClick={handleEditClick}
            />
            <Button
              icon={<DocumentDuplicateIcon className="h-4 w-5" />}
              tooltip="복사"
              tooltipOptions={{ position: 'top' }}
              className="!text-gray-500 w-fit"
              onClick={handleCopy}
            />
          </div>
        )}

        {/* 챗봇 메시지일 경우: 복사, 재생성 버튼 */}
        {message.sender === 'chatbot' && (
          <div className="flex gap-3 text-xs text-gray-500 mt-2">
            <Button
              icon={<DocumentDuplicateIcon className="h-4 w-5" />}
              tooltip="복사"
              tooltipOptions={{ position: 'top' }}
              className="!text-gray-500 w-fit"
              onClick={handleCopy}
            />
            <Button
              icon={<ArrowPathIcon className="h-4 w-5" />}
              className="!text-gray-500 w-fit"
              tooltipOptions={{ position: 'top' }}
              tooltip="답변 재생성"
              onClick={handleRegenerateClick}
            />
          </div>
        )}
      </div>
    </div>
  );
}