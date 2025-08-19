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
import LoadingBubble from './LoadingBubble';

type MessageType = 'text' | 'recommendation' | 'flightinfo' | 'edit' | 'again';

interface WebSocketMessageDto {
  messageId: string; // ✨ 사용자 메시지의 UUID를 담을 필드
  sessionId: string;
  content: string;
  messageType: MessageType;
  parentId: string | null;
}

export interface WebSocketResponseDto {
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
  // 부모로부터 초기 채팅 내역을 받을 prop
  initialHistory: WebSocketResponseDto[];
}

export default function ChatBotScreen({ sessionId, initialHistory  }: ChatBotScreenProps) {
  const stompClientRef = useRef<Client | null>(null);
  // 채팅 메시지 상태의 초기값을 부모에게서 받은 initialHistory로 설정
  const [chatMessages, setChatMessages] = useState<WebSocketResponseDto[]>(initialHistory);
  const [messageInputValue, setMessageInputValue] = useState('');
  const [flightNumberInputValue, setFlightNumberInputValue] = useState('');
  const [recommendedQuestions, setRecommendedQuestions] = useState<string[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isBotReplying, setIsBotReplying] = useState(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null);

  useEffect(() => {
    if (chatContainerRef.current) {
      // scrollHeight는 스크롤 가능한 전체 높이를 의미합니다.
      // scrollTop을 scrollHeight로 설정하여 스크롤을 맨 아래로 내립니다.
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatMessages, isBotReplying]);

  /**
   * 부모로부터 받은 initialHistory가 변경될 때마다(예: 로그아웃) 
   * 화면의 채팅 메시지 목록을 업데이트하기 위한 useEffect를 추가합니다.
   */
  useEffect(() => {
    setChatMessages(initialHistory);
  }, [initialHistory]);

  // 웹소켓 연결 및 구독 로직
  useEffect(() => {
    if (!sessionId) return;
    const client = new Client({
      webSocketFactory: () => new SockJS(`${process.env.NEXT_PUBLIC_API_BASE_URL}/ws-chat`),
      connectHeaders: { Authorization: `Bearer ${localStorage.getItem('jwt_token') || ''}` },
      onConnect: () => {
        setIsConnected(true);
        
        // ✅ 서버로부터 메시지를 받는 구독 로직을 수정합니다.
        client.subscribe(`/topic/chat/${sessionId}`, (message) => {
          const receivedMessage: WebSocketResponseDto = JSON.parse(message.body);
          
          setIsBotReplying(false); // 로딩 종료

          if (receivedMessage.messageType === 'recommendation') {
            setRecommendedQuestions(receivedMessage.content.split(';'));
          } else {
            // 챗봇의 답변이고, 이 답변이 어떤 사용자 질문에 대한 것인지 식별 가능할 때 (수정/재생성)
            if (receivedMessage.sender === 'chatbot' && receivedMessage.userMessageId) {
              setChatMessages(prevMessages => {
                // 기존 대화 목록에서, 동일한 사용자 질문에 대한 챗봇의 이전 답변을 찾습니다.
                const oldBotMessageIndex = prevMessages.findIndex(
                  msg => msg.sender === 'chatbot' && msg.userMessageId === receivedMessage.userMessageId
                );

                if (oldBotMessageIndex !== -1) {
                  // ✨ 만약 이전 답변을 찾았다면, 그 답변을 새로 받은 메시지로 '교체'합니다.
                  const updatedMessages = [...prevMessages];
                  updatedMessages[oldBotMessageIndex] = receivedMessage;
                  return updatedMessages;
                } else {
                  // 이전 답변을 찾지 못했다면 (첫 답변), 그냥 목록에 추가합니다.
                  return [...prevMessages, receivedMessage];
                }
              });
            } else {
              // 사용자 메시지이거나, userMessageId가 없는 일반 챗봇 메시지는 그냥 추가합니다.
              setChatMessages((prevMessages) => [...prevMessages, receivedMessage]);
            }
          }
        });
      },
      onStompError: (frame) => {
        console.error('❌ STOMP 오류:', frame.headers['message']);
        setIsBotReplying(false);
      },
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

    // 메시지를 보낸 직후, 로딩 상태를 시작합니다.
    setIsBotReplying(true);
  };
  
  // 1. 새 메시지 전송 (parentId: null)
  const handleSendNewMessage = (content: string, type: 'text' | 'flightinfo' | 'recommendation') => {
    if (!content.trim() || !sessionId) return;
    
    // ✨ 프론트엔드에서 UUID를 직접 생성합니다.
    const newUuid = crypto.randomUUID();

    const userMessage: WebSocketResponseDto = {
        // ✨ 생성한 UUID를 사용합니다.
        messageId: newUuid,
        userMessageId: null,
        sessionId: sessionId,
        sender: 'user',
        content: content.trim(),
        messageType: 'text',
        createdAt: new Date().toISOString(),
    };

    setChatMessages((prev) => [...prev, userMessage]);
    
    publishMessage({
        // ✨ 생성한 UUID를 백엔드로 전송합니다.
        messageId: newUuid,
        sessionId,
        content: content.trim(),
        messageType: type,
        parentId: null,
    });

    setMessageInputValue('');
    setFlightNumberInputValue('');
    setRecommendedQuestions([]);
  };

// 이 함수는 수정 내용을 최종 '저장(커밋)'하는 역할을 합니다.
const handleCommitEdit = (originalMessageId: string, newContent: string) => {
    if (!newContent.trim() || !sessionId) return;

    const newEditUuid = crypto.randomUUID();

    setChatMessages(prevMessages => {
        const filteredMessages = prevMessages.filter(
            msg => !(msg.sender === 'chatbot' && msg.userMessageId === originalMessageId)
        );
        const updatedMessages = filteredMessages.map(msg =>
            msg.messageId === originalMessageId
                ? { ...msg, content: newContent.trim() }
                : msg
        );
        return updatedMessages;
    });

    publishMessage({
      messageId: newEditUuid,
      sessionId,
      content: newContent.trim(),
      messageType: 'edit',
      parentId: originalMessageId,
    });
    
    setEditingMessageId(null);
  };

  // 답변 재생성 (parentId: 원본 질문 ID)
  const handleRegenerateAnswer = (originalUserMessageId: string) => {
    if (!sessionId) return;

    setChatMessages(prevMessages => 
      prevMessages.filter(msg => !(msg.sender === 'chatbot' && msg.userMessageId === originalUserMessageId))
    );

    publishMessage({
      // ✨ 규칙: 재생성 요청의 messageId와 parentId는 동일하게 원본 질문 ID로 설정합니다.
      messageId: originalUserMessageId, 
      sessionId,
      content: '', // 내용은 비워도 됨
      messageType: 'again',
      parentId: originalUserMessageId,
    });
  };

  const handleStartEdit = (messageId: string) => {
    setEditingMessageId(messageId);
  };

  const handleCancelEdit = () => {
    setEditingMessageId(null);
  };

  const handleMessageInputChange = (e: React.ChangeEvent<HTMLInputElement>) => setMessageInputValue(e.target.value);
  const handleMessageInputSend = () => handleSendNewMessage(messageInputValue, 'text');
  const handleFlightNumberInputSend = () => handleSendNewMessage(flightNumberInputValue, 'flightinfo');
  const handleRecommendedQuestionClick = (question: string) => handleSendNewMessage(question, 'recommendation');

  // 배열의 마지막 요소 인덱스를 찾는 헬퍼 함수
  const findLastIndex = <T,>(array: T[], predicate: (value: T, index: number, obj: T[]) => boolean): number => {
    let l = array.length;
    while (l--) {
      if (predicate(array[l], l, array)) return l;
    }
    return -1;
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
        <div ref={chatContainerRef} className="flex-1 overflow-y-auto p-4 space-y-4">
          {(() => {
            // 렌더링 전에 마지막 사용자 메시지와 챗봇 메시지의 인덱스를 찾습니다.
            const lastUserMessageIndex = findLastIndex(chatMessages, msg => msg.sender === 'user');
            const lastBotMessageIndex = findLastIndex(chatMessages, msg => msg.sender === 'chatbot');

            return chatMessages.map((msg, index) => {
              // 2. 현재 메시지가 각 타입의 마지막 메시지인지 판별합니다.
              const isLastUserMessage = msg.sender === 'user' && index === lastUserMessageIndex;
              const isLastBotMessage = msg.sender === 'chatbot' && index === lastBotMessageIndex;

              return (
                <ChatBubble
                  key={msg.messageId}
                  message={msg}
                  isLastUserMessage={isLastUserMessage}
                  isLastBotMessage={isLastBotMessage}
                  isEditing={editingMessageId === msg.messageId}
                  isBotReplying={isBotReplying}
                  onStartEdit={handleStartEdit}
                  onCommitEdit={handleCommitEdit}
                  onCancelEdit={handleCancelEdit}
                  onRegenerate={handleRegenerateAnswer}
                />
              );
            });
          })()}

          {isBotReplying && <LoadingBubble />}
          {recommendedQuestions.length > 0 && !isBotReplying && (
            <RecommendedQuestions 
              questions={recommendedQuestions} 
              onQuestionClick={handleRecommendedQuestionClick} 
            />
          )}
        </div>
      )}

      {/* 하단 텍스트 박스 (SearchInput 재사용) */}
      <div className="absolute bottom-0 left-0 right-0 p-4 bg-blue-50 shadow-md">
        <SearchInput
          placeholder="무엇이든 물어보세요!"
          value={messageInputValue}
          onChange={handleMessageInputChange}
          onSend={handleMessageInputSend}
          disabled={!isConnected || isBotReplying} 
        />
      </div>
    </div>
  );
}