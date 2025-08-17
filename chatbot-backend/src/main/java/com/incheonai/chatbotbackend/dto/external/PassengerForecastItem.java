// com/incheonai/chatbotbackend/dto/external/PassengerForecastItem.java
package com.incheonai.chatbotbackend.dto.external;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@JsonIgnoreProperties(ignoreUnknown = true)
public class PassengerForecastItem {
    private String adate; // 표출일자
    private String atime; // 시간대
    // T1 입국장 (개별)
    private Double t1sum1; // T1 입국장 동편(A,B)
    private Double t1sum2; // T1 입국장 서편(E,F)
    private Double t1sum3; // T1 입국심사(C)
    private Double t1sum4; // T1 입국심사(D)

    // T1 출국장 (개별)
    private Double t1sum5; // T1 출국장1,2
    private Double t1sum6; // T1 출국장3
    private Double t1sum7; // T1 출국장4
    private Double t1sum8; // T1 출국장5,6

    // T1 합계
    private Double t1sumset1; // T1 입국장 합계
    private Double t1sumset2; // T1 출국장 합계

    // T2 입국장 (개별)
    private Double t2sum1; // T2 입국장 1
    private Double t2sum2; // T2 입국장 2

    // T2 출국장 (개별)
    private Double t2sum3; // T2 출국장 1
    private Double t2sum4; // T2 출국장 2

    // T2 합계
    private Double t2sumset1; // T2 입국장 합계
    private Double t2sumset2; // T2 출국장 합계
}