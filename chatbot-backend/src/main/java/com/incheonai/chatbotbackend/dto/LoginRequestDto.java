package com.incheonai.chatbotbackend.dto;

import com.incheonai.chatbotbackend.domain.jpa.LoginProvider;
import jakarta.validation.constraints.NotBlank;

/**
 * 로그인 요청 시 사용되는 DTO
 * @param userId 사용자 아이디
 * @param password 사용자 비밀번호
 */
public record LoginRequestDto(
        @NotBlank(message = "아이디를 입력해주세요.")
        String userId,

        @NotBlank(message = "비밀번호를 입력해주세요.")
        String password
) {}