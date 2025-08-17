// com/incheonai/chatbotbackend/dto/external/FlightDepartureInfoItem.java
package com.incheonai.chatbotbackend.dto.external;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@JsonIgnoreProperties(ignoreUnknown = true)
public class FlightDepartureInfoItem {
    private String airline; // 항공사
    private String airport; // 도착지공항명
    private String airportCode; // 공항코드
    private String chkinrange; // 체크인카운터
    private String estimatedDateTime; // 변경일시
    private String flightId; // 편명
    private String gatenumber; // 탑승구
    private String remark; // 현황
    private String scheduleDateTime; // 예정일시
    private String terminalid; // 터미널
}