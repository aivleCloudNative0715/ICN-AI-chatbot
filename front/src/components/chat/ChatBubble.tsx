// src/components/chat/ChatBubble.tsx
import React from 'react';
import { Button } from 'primereact/button';
import { PencilIcon, DocumentDuplicateIcon, ArrowPathIcon } from '@heroicons/react/24/outline';

interface ChatBubbleProps {
  message: string;
  isUser: boolean;
}

export default function ChatBubble({ message, isUser }: ChatBubbleProps) {
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div
        className={`relative max-w-lg p-3 rounded-xl shadow-md ${
          isUser
            ? 'bg-blue-200 text-gray-800 rounded-br-none'
            : 'bg-white text-gray-800 rounded-bl-none'
        }`}
      >
        <p className="text-sm sm:text-base">{message}</p>
        <div className={`absolute ${isUser ? 'right-0 -bottom-6' : 'left-0 -bottom-6'} flex gap-1 text-xs text-gray-500`}>
          {isUser ? (
            // 사용자 말풍선 하단 버튼 (UI-09-25014)
            <>
              <Button
                icon={<PencilIcon className="h-3 w-3" />}
                label="편집"
                className="p-button-text p-button-sm !text-gray-500 hover:!bg-gray-200"
                pt={{
                  root: { className: '!p-1 flex items-center !min-w-fit' },
                  label: { className: '!text-xs' },
                  icon: { className: '!mr-1' },
                }}
                onClick={() => console.log('편집 클릭')}
              />
              <Button
                icon={<DocumentDuplicateIcon className="h-3 w-3" />}
                label="복사"
                className="p-button-text p-button-sm !text-gray-500 hover:!bg-gray-200"
                pt={{
                  root: { className: '!p-1 flex items-center !min-w-fit' },
                  label: { className: '!text-xs' },
                  icon: { className: '!mr-1' },
                }}
                onClick={() => navigator.clipboard.writeText(message)}
              />
            </>
          ) : (
            // 챗봇 말풍선 하단 버튼 (UI-09-25015)
            <>
              <Button
                icon={<DocumentDuplicateIcon className="h-3 w-3" />}
                label="답변 복사"
                className="p-button-text p-button-sm !text-gray-500 hover:!bg-gray-200"
                pt={{
                  root: { className: '!p-1 flex items-center !min-w-fit' },
                  label: { className: '!text-xs' },
                  icon: { className: '!mr-1' },
                }}
                onClick={() => navigator.clipboard.writeText(message)}
              />
              <Button
                icon={<ArrowPathIcon className="h-3 w-3" />}
                label="재대답"
                className="p-button-text p-button-sm !text-gray-500 hover:!bg-gray-200"
                pt={{
                  root: { className: '!p-1 flex items-center !min-w-fit' },
                  label: { className: '!text-xs' },
                  icon: { className: '!mr-1' },
                }}
                onClick={() => console.log('재대답 클릭')}
              />
            </>
          )}
        </div>
      </div>
    </div>
  );
}