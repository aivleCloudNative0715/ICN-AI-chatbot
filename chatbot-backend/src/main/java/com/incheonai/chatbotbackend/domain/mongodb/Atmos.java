package com.incheonai.chatbotbackend.domain.mongodb;

import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;

import lombok.Getter;
import lombok.ToString;

import java.time.Instant;

@Getter
@ToString
@Document(collection = "ATMOS") // 실제 컬렉션 이름 지정
public class Atmos {

    @Id
    private String id; // MongoDB의 _id 필드

    @Field("tm")
    private Instant time;

    @Field("ta") // 온도
    private String temperature;

    @Field("hm")
    private String humidity;

    @Field("rn")
    private String rainAmount;

    @Field("ws_10")
    private String windSpeed;

    // 필요한 다른 필드들도 추가...
}