// com/incheonai/chatbotbackend/dto/external/ParkingInfoItem.java
package com.incheonai.chatbotbackend.dto.external;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@JsonIgnoreProperties(ignoreUnknown = true)
public class ParkingInfoItem {
    private String floor; // 주차장 구분
    private String parking; // 주차구역별 주차대수
    private String parkingarea; // 전체 주차 면 수 필드
    private String datetm; // 주차 현황 시간
}