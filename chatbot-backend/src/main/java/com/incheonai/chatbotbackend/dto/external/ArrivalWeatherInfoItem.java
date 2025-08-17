// com/incheonai/chatbotbackend/dto/external/ArrivalWeatherInfoItem.java
package com.incheonai.chatbotbackend.dto.external;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@JsonIgnoreProperties(ignoreUnknown = true)
public class ArrivalWeatherInfoItem {
    private String airline; // 항공사
    private String flightId; // 편명
    private String airport; // 출발 공항
    private String temp; // 관측 기온
    private String wimage; // 날씨 이미지 URL
}