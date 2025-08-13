// src/components/chat/ChatBubble.tsx
import React, { useEffect, useState } from 'react';
import { Button } from 'primereact/button';
import { PencilIcon, DocumentDuplicateIcon, ArrowPathIcon, CheckIcon, XMarkIcon } from '@heroicons/react/24/outline';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// ChatBubble이 받을 props 타입
interface ChatBubbleProps {
  message: {
    messageId: string;
    content: string;
    sender: 'user' | 'chatbot';
    userMessageId: string | null;
  };
  isLastUserMessage: boolean;
  isLastBotMessage: boolean;
  isEditing: boolean;
  isBotReplying: boolean;
  onStartEdit: (messageId: string) => void;
  onCommitEdit: (originalMessageId: string, newContent: string) => void;
  onCancelEdit: () => void;
  onRegenerate: (originalUserMessageId: string) => void;
}

export default function ChatBubble({
  message,
  isLastUserMessage,
  isLastBotMessage,
  isEditing,
  isBotReplying, // prop 받기
  onStartEdit,
  onCommitEdit,
  onCancelEdit,
  onRegenerate,
}: ChatBubbleProps) {

  const [editedContent, setEditedContent] = useState(message.content);

  useEffect(() => {
    // message.content가 외부에서 변경될 경우(예: 스트리밍 응답)를 대비해 동기화
    setEditedContent(message.content);
  }, [message.content]);

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
  };
  
  const handleRegenerateClick = () => {
    if (message.userMessageId) {
       onRegenerate(message.userMessageId);
    }
  };

  // isEditing 값에 따라 다른 UI를 렌더링합니다.
  if (isEditing) {
    // --- 편집 모드 UI ---
    return (
      <div className="flex justify-end mb-4">
        <div className="flex flex-col w-full max-w-lg p-3 rounded-xl bg-blue-200 shadow-md">
          <textarea
            value={editedContent}
            onChange={(e) => setEditedContent(e.target.value)}
            className="w-full p-2 border border-blue-300 rounded-md bg-white text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
            rows={4}
            autoFocus
          />
          <div className="flex justify-end gap-2 mt-2">
            <Button
              icon={<CheckIcon className="h-4 w-4" />}
              label="저장"
              className="p-button-sm"
              onClick={() => onCommitEdit(message.messageId, editedContent)}
            />
            <Button
              icon={<XMarkIcon className="h-4 w-4" />}
              label="취소"
              className="p-button-sm p-button-secondary"
              onClick={onCancelEdit}
            />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'} mb-4`}>
      <div
        className={`flex flex-col max-w-lg p-3 rounded-xl ${
          message.sender === 'user'
            ? 'bg-blue-200 text-gray-800 rounded-br-none items-end shadow-md'
            : 'bg-blue-50 text-gray-800 rounded-bl-none'
        }`}
      >
        <div className="prose max-w-none text-sm sm:text-base">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {message.content}
          </ReactMarkdown>
        </div>

        {/* 버튼 영역 */}
        <div className="flex gap-3 text-xs text-gray-500 mt-2">
          {/* '수정' 버튼은 isLastUserMessage일 때만 표시 */}
          {message.sender === 'user' && isLastUserMessage && (
            <Button
              icon={<PencilIcon className="h-4 w-4" />}
              tooltip="편집"
              className="!text-gray-500 w-fit"
              onClick={() => onStartEdit(message.messageId)}
              disabled={isBotReplying}
            />
          )}

          {/* '재생성' 버튼은 isLastBotMessage일 때만 표시 */}
          {message.sender === 'chatbot' && isLastBotMessage && (
            <Button
              icon={<ArrowPathIcon className="h-4 w-5" />}
              className="!text-gray-500 w-fit"
              tooltipOptions={{ position: 'top' }}
              tooltip="답변 재생성"
              onClick={handleRegenerateClick}
              disabled={isBotReplying}
            />
          )}

          {/* 복사 버튼은 항상 표시 */}
          <Button
            icon={<DocumentDuplicateIcon className="h-4 w-5" />}
            tooltip="복사"
            tooltipOptions={{ position: 'top' }}
            className="!text-gray-500 w-fit"
            onClick={handleCopy}
          />
        </div>
      </div>
    </div>
  );
}