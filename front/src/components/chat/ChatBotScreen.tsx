// src/components/chat/ChatBotScreen.tsx
'use client';

import React, { useState } from 'react';
import Image from 'next/image';
import SearchInput from '@/components/common/SearchInput';
import { PaperAirplaneIcon } from '@heroicons/react/24/outline'; // 화살표 아이콘
import ChatBubble from '@/components/chat/ChatBubble'; // ChatBubble 컴포넌트 임포트
import RecommendedQuestions from '@/components/chat/RecommendedQuestions'; // RecommendedQuestions 컴포넌트 임포트

interface ChatBotScreenProps {
  isLoggedIn: boolean; // 로그인 상태 prop
  onLoginStatusChange: (status: boolean) => void; // 로그인 상태 변경 핸들러
  onSidebarToggle: () => void; // 사이드바 토글 핸들러
}

export default function ChatBotScreen({
  isLoggedIn,
  onLoginStatusChange,
  onSidebarToggle,
}: ChatBotScreenProps) {
  // 채팅 메시지 목록 상태 관리
  const [chatMessages, setChatMessages] = useState<
    { id: number; sender: 'user' | 'bot'; content: string; type?: string }[]
  >([]);
  const [messageInputValue, setMessageInputValue] = useState(''); // 메시지 입력 필드 상태
  const [flightNumberInputValue, setFlightNumberInputValue] = useState(''); // 편명 입력 필드 상태
  const [recommendedQuestions, setRecommendedQuestions] = useState<string[]>([]); // 추천 질문 상태 추가

  const handleSendMessage = (message: string, type?: string) => {
    if (message.trim()) {
      const newUserMessage = {
        id: chatMessages.length + 1,
        sender: 'user' as const,
        content: message.trim(),
        type: type,
      };
      setChatMessages((prevMessages) => [...prevMessages, newUserMessage]);
      setMessageInputValue(''); // 메시지 입력 필드 초기화
      setFlightNumberInputValue(''); // 편명 입력 필드도 초기화 (선택 사항, 필요에 따라 유지 가능)
      setRecommendedQuestions([]); // 새 메시지 전송 시 추천 질문 초기화

      // TODO: 여기에 챗봇 API 호출 로직 추가 (API-09-25031)
      // 챗봇 답변을 시뮬레이션
      setTimeout(() => {
        const botResponse = {
          id: chatMessages.length + 2,
          sender: 'bot' as const,
          content: `"${message}"에 대한 챗봇의 답변입니다.`,
        };
        setChatMessages((prevMessages) => [...prevMessages, botResponse]);

        // 챗봇 답변 후 추천 질문 설정 (예시 로직)
        if (message.includes('편명') || message.includes('항공')) {
          setRecommendedQuestions(['출국 절차 안내', '입국 절차 안내', '수하물 규정']);
        } else if (message.includes('수하물')) {
          setRecommendedQuestions(['기내 수하물 규정', '위탁 수하물 규정', '초과 수하물 요금']);
        } else {
          setRecommendedQuestions(['다른 질문은 없으신가요?', '가장 인기 있는 질문은?']);
        }
      }, 500);
    }
  };

  const handleMessageInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setMessageInputValue(e.target.value);
  };

  const handleMessageInputSend = () => {
    handleSendMessage(messageInputValue);
  };

  const handleFlightNumberInputSend = () => {
    handleSendMessage(flightNumberInputValue, 'flightNumber'); // 편명 입력 메시지 전송
  };

  const handleRecommendedQuestionClick = (question: string) => {
    handleSendMessage(question, 'Recommendation'); // 추천 질문 클릭 시 type을 'Recommendation'으로 설정
  };

  // 하단 SearchInput의 높이를 고려하여 padding-bottom을 설정 (예시: 80px 또는 p-20)
  const paddingBottomClass = 'pb-20'; // 대략적인 SearchInput 높이에 맞춰 여유 공간 확보

  return (
    <div className={`relative flex flex-col flex-1 h-full bg-blue-50 ${paddingBottomClass}`}>
      {/* 챗봇 아이콘 및 인사말 (채팅 기록이 없을 때만 표시)*/}
      {chatMessages.length === 0 && (
        <div className="flex flex-col items-center justify-center w-full flex-grow">
          <Image
            src="/airplane-icon.png"
            alt="Airplane Icon"
            width={150}
            height={150}
            className="mb-6"
          />
          <h1 className="text-2xl font-semibold text-gray-800 mb-2 text-center">
            인천공항 AI 챗봇 서비스입니다! 궁금한 점을 물어봐주세요!
          </h1>
          <p className="text-gray-600 mb-8 text-center">
            편명 입력 시 더 자세한 답변이 가능합니다.
          </p>
          {/* 편명 입력 텍스트 박스 - SearchInput을 사용하지 않고 직접 구현 */}
          <div className="relative flex items-center justify-center w-full max-w-sm px-4 py-3 border-b-2 border-gray-300 text-gray-700 placeholder-gray-400 focus-within:border-blue-500 transition-all duration-300">
            <span className="mr-2 text-gray-500">
              <PaperAirplaneIcon className="h-6 w-6" />
            </span>
            <input
              type="text"
              placeholder="편명 입력"
              className="flex-grow bg-transparent outline-none text-center"
              value={flightNumberInputValue} // 별도의 상태 변수 사용
              onChange={(e) => setFlightNumberInputValue(e.target.value)} // 별도의 핸들러
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  handleFlightNumberInputSend(); // 엔터 시 편명으로 전송
                }
              }}
            />
          </div>
        </div>
      )}

      {/* 채팅 메시지 표시 영역 */}
      {chatMessages.length > 0 && (
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {chatMessages.map((message) => (
            <ChatBubble
              key={message.id}
              message={message.content}
              isUser={message.sender === 'user'}
            />
          ))}
          {/* 추천 질문 표시 */}
          {recommendedQuestions.length > 0 && (
            <RecommendedQuestions questions={recommendedQuestions} onQuestionClick={handleRecommendedQuestionClick} />
          )}
        </div>
      )}

      {/* 하단 텍스트 박스 (SearchInput 재사용) */}
      <div className="absolute bottom-0 left-0 right-0 p-4 bg-blue-50 shadow-md">
        <SearchInput
          placeholder="무엇이든 물어보세요!"
          value={messageInputValue} // 메시지 입력 필드 상태 사용
          onChange={handleMessageInputChange} // 메시지 입력 핸들러
          onSend={handleMessageInputSend} // 메시지 전송 핸들러
        />
      </div>
    </div>
  );
}