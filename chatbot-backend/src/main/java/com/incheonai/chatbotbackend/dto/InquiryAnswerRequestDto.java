package com.incheonai.chatbotbackend.dto;

/**
 * 문의 답변 생성/수정 요청 DTO
 * - AdminController#addAnswer, updateAnswer 에서 사용
 *
 * @param content  답변 본문 내용
 * @param adminId  답변자(관리자) 아이디
 */
public record InquiryAnswerRequestDto(
        String content,
        String adminId
) {

}