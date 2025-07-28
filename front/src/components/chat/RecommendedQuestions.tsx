// src/components/chat/RecommendedQuestions.tsx
import React from 'react';
import { Button } from 'primereact/button';

interface RecommendedQuestionsProps {
  questions: string[];
  onQuestionClick: (question: string) => void;
}

export default function RecommendedQuestions({ questions, onQuestionClick }: RecommendedQuestionsProps) {
  if (questions.length === 0) {
    return null;
  }

  return (
    <div className="flex flex-col items-start mt-4 space-y-2">
      {questions.map((question, index) => (
        <Button
          key={index}
          label={question}
          className="w-full max-w-sm py-2 px-4 rounded-full border border-primary bg-blue-50 text-primary hover:bg-blue-100 transition-colors duration-200 text-sm font-semibold"
          onClick={() => onQuestionClick(question)}
          pt={{
            root: {
              className: '!shadow-none'
            },
            label: {
              className: 'whitespace-nowrap' // 텍스트 줄바꿈 방지
            }
          }}
        />
      ))}
    </div>
  );
}