// com/incheonai/chatbotbackend/dto/external/PassengerForecastItem.java
package com.incheonai.chatbotbackend.dto.external;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@JsonIgnoreProperties(ignoreUnknown = true)
public class PassengerForecastItem {
    // 시간대, 날짜, 터미널별 입/출국장 예상 승객 수 합계
    private String adate;
    private String atime;
    private Double  t1sumset1;
    private Double  t1sumset2;
    private Double  t2sumset1;
    private Double  t2sumset2;
}