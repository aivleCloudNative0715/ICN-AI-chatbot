package com.incheonai.chatbotbackend.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor // 모든 필드를 포함하는 생성자를 만듭니다.
public class TemperatureResponseDto {

    private Double temperature;
    private String timestamp;
}