package com.incheonai.chatbotbackend.dto;

import com.incheonai.chatbotbackend.domain.jpa.AdminRole;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
public class AdminCreateRequestDto {
    private String adminId;
    private String password;
    private String adminName;
    private AdminRole role;
}
