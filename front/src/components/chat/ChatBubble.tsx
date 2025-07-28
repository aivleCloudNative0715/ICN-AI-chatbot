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
        className={`flex flex-col max-w-lg p-3 rounded-xl ${
          isUser
            ? 'bg-blue-200 text-gray-800 rounded-br-none items-end shadow-md'
            : 'bg-blue-50 text-gray-800 rounded-bl-none'
        }`}
      >
        <p className="text-sm sm:text-base mb-2">{message}</p>

        {isUser && (
          <div className="flex gap-3 text-xs text-gray-500 mt-2">
            <Button
              icon={<PencilIcon className="h-4 w-4" />}
              tooltip="편집"
              tooltipOptions={{ position: 'top' }}
              className="!text-gray-500 w-fit"
              onClick={() => console.log('편집 클릭')}
            />
            <Button
              icon={<DocumentDuplicateIcon className="h-4 w-5" />}
              tooltip="복사"
              tooltipOptions={{ position: 'top' }}
              className="!text-gray-500 w-fit"
              onClick={() => navigator.clipboard.writeText(message)}
            />
          </div>
        )}

        {!isUser && (
          <div className="flex gap-3 text-xs text-gray-500 mt-2">
            <Button
              icon={<DocumentDuplicateIcon className="h-4 w-5" />}
              tooltip="복사"
              tooltipOptions={{ position: 'top' }}
              className="!text-gray-500 w-fit"
              onClick={() => navigator.clipboard.writeText(message)}
            />
            <Button
              icon={<ArrowPathIcon className="h-4 w-5" />}
              className="!text-gray-500 w-fit"
              tooltipOptions={{ position: 'top' }}
              tooltip="대답 재생성"
              onClick={() => console.log('재대답 클릭')}
            />
          </div>
        )}
      </div>
    </div>
  );
}
