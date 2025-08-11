// src/components/chat/ChatBotScreen.tsx
'use client';

import React, { useEffect, useRef, useState } from 'react';
import Image from 'next/image';
import SearchInput from '@/components/common/SearchInput';
import { PaperAirplaneIcon } from '@heroicons/react/24/outline'; // 화살표 아이콘
import ChatBubble from '@/components/chat/ChatBubble'; // ChatBubble 컴포넌트 임포트
import RecommendedQuestions from '@/components/chat/RecommendedQuestions'; // RecommendedQuestions 컴포넌트 임포트
// STOMP 관련 라이브러리 임포트
import { Client } from '@stomp/stompjs';
import SockJS from 'sockjs-client';
import { API_BASE_URL } from '@/lib/api';

type MessageType = 'text' | 'recommendation' | 'flightinfo' | 'edit' | 'again';

interface WebSocketMessageDto {
  sessionId: string;
  content: string;
  messageType: MessageType;
  parentId: string | null;
}

interface WebSocketResponseDto {
  messageId: string;
  userMessageId: string | null;
  sessionId: string;
  sender: 'user' | 'chatbot';
  content: string;
  messageType: 'text' | 'recommendation' | 'again';
  createdAt: string;
}

interface ChatBotScreenProps {
  isLoggedIn: boolean;
  sessionId: string | null;
}

export default function ChatBotScreen({ sessionId }: ChatBotScreenProps) {
  const stompClientRef = useRef<Client | null>(null);
  const [chatMessages, setChatMessages] = useState<WebSocketResponseDto[]>([]);
  const [messageInputValue, setMessageInputValue] = useState('');
  const [flightNumberInputValue, setFlightNumberInputValue] = useState('');
  const [recommendedQuestions, setRecommendedQuestions] = useState<string[]>([]);
  const [isConnected, setIsConnected] = useState(false);

  // 웹소켓 연결 및 구독 로직
  useEffect(() => {
    if (!sessionId) return;
    const client = new Client({
      webSocketFactory: () => new SockJS(`${API_BASE_URL}/ws-chat`),
      connectHeaders: { Authorization: `Bearer ${localStorage.getItem('jwt_token') || ''}` },
      onConnect: () => {
        setIsConnected(true);
        console.log('✅ STOMP: 연결 성공');
        
        // 서버로부터 메시지를 받는 구독 로직 수정
        client.subscribe(`/topic/chat/${sessionId}`, (message) => {
          const receivedMessage: WebSocketResponseDto = JSON.parse(message.body);
          console.log('📥 STOMP: 메시지 수신', receivedMessage);

          if (receivedMessage.messageType === 'recommendation') {
            // 추천 질문은 화면에 그리지 않고, 추천 질문 목록 상태만 업데이트
            setRecommendedQuestions(receivedMessage.content.split(';'));
          } else {
            // 그 외 모든 메시지(사용자 질문 포함)는 chatMessages 상태에 추가
            setChatMessages((prevMessages) => [...prevMessages, receivedMessage]);
          }
        });
      },
      onStompError: (frame) => console.error('❌ STOMP 오류:', frame.headers['message']),
    });

    client.activate();
    stompClientRef.current = client;

    return () => {
      client.deactivate();
      setIsConnected(false);
    };
  }, [sessionId]);

  // 공통 발신 함수
  const publishMessage = (dto: WebSocketMessageDto) => {
    if (!stompClientRef.current || !isConnected) {
      alert('연결이 불안정합니다. 잠시 후 다시 시도해주세요.');
      return;
    }
    stompClientRef.current.publish({
      destination: '/app/chat.sendMessage',
      body: JSON.stringify(dto),
    });
    console.log('📤 STOMP: 메시지 발신', dto);
  };
  
  // 1. 새 메시지 전송 (parentId: null)
  const handleSendNewMessage = (content: string, type: 'text' | 'flightinfo' | 'recommendation') => {
    if (!content.trim() || !sessionId) return;
    
    // --- 1. 사용자 메시지를 화면에 즉시 표시하기 위한 객체 생성 ---
    const userMessage: WebSocketResponseDto = {
        messageId: `local-user-${Date.now()}`, // 임시 로컬 ID 부여
        userMessageId: null,
        sessionId: sessionId,
        sender: 'user',
        content: content.trim(),
        messageType: 'text', // 사용자 메시지는 항상 'text'로 화면에 표시
        createdAt: new Date().toISOString(),
    };

    // --- 2. 생성한 객체를 chatMessages 상태에 바로 추가 ---
    setChatMessages((prev) => [...prev, userMessage]);
    
    // --- 3. 서버로 메시지 전송 (기존 로직) ---
    publishMessage({
        sessionId,
        content: content.trim(),
        messageType: type,
        parentId: null,
    });

    setMessageInputValue('');
    setFlightNumberInputValue('');
    setRecommendedQuestions([]);
  };

  // 2. 질문 수정 (parentId: 원본 질문 ID)
  const handleEditMessage = (originalMessageId: string, newContent: string) => {
    if (!newContent.trim() || !sessionId) return;
    
    publishMessage({
      sessionId,
      content: newContent.trim(),
      messageType: 'edit',
      parentId: originalMessageId, // 수정할 원본 질문 ID
    });
  };

  // 3. 답변 재생성 (parentId: 원본 질문 ID)
  const handleRegenerateAnswer = (originalUserMessageId: string) => {
    if (!sessionId) return;

    publishMessage({
      sessionId,
      content: '', // 내용은 비워도 됨
      messageType: 'again',
      parentId: originalUserMessageId, // 답변을 다시 받을 원본 질문 ID
    });
  };

  const handleMessageInputChange = (e: React.ChangeEvent<HTMLInputElement>) => setMessageInputValue(e.target.value);
  const handleMessageInputSend = () => handleSendNewMessage(messageInputValue, 'text');
  const handleFlightNumberInputSend = () => handleSendNewMessage(flightNumberInputValue, 'flightinfo');
  const handleRecommendedQuestionClick = (question: string) => handleSendNewMessage(question, 'recommendation');
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
          {chatMessages.map((msg) => (
            <ChatBubble
              key={msg.messageId} // 고유 키는 messageId 사용
              message={{
                 messageId: msg.messageId,
                 content: msg.content,
                 sender: msg.sender,
                 userMessageId: msg.userMessageId
              }}
              onEdit={handleEditMessage}
              onRegenerate={handleRegenerateAnswer}
            />
          ))}
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
          disabled={!isConnected} //  연결 안됐으면 입력 비활성화
        />
      </div>
    </div>
  );
}