package com.incheonai.chatbotbackend.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

// 회원가입 요청 데이터를 담을 DTO
public record SignUpRequestDto(
        @NotBlank(message = "아이디를 입력해주세요.")
        @Size(min = 6, max = 12, message = "아이디는 6자 이상 12자 이하로 입력해주세요.")
        String userId,

        @NotBlank(message = "비밀번호를 입력해주세요.")
        @Size(min = 10, max=20, message = "비밀번호는 8자 이상 20자 이하로 입력해주세요.")
        String password
) {}