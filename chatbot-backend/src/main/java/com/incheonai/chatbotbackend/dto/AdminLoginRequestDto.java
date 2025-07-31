package com.incheonai.chatbotbackend.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
public class AdminLoginRequestDto {
    private String adminId;
    private String password;
}
