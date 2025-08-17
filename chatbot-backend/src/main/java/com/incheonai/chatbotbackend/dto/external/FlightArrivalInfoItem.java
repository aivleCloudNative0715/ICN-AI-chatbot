// com/incheonai/chatbotbackend/dto/external/FlightArrivalInfoItem.java
package com.incheonai.chatbotbackend.dto.external;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@JsonIgnoreProperties(ignoreUnknown = true)
public class FlightArrivalInfoItem {
    private String airline; // 항공사
    private String airport; // 출발공항명
    private String airportCode; // 공항코드
    private String carousel; // 수하물수취대
    private String estimatedDateTime; // 변경일시
    private String exitnumber; // 출구
    private String flightId; // 편명
    private String gatenumber; // 탑승구
    private String remark; // 현황
    private String scheduleDateTime; // 예정일시
    private String terminalid; // 터미널
}