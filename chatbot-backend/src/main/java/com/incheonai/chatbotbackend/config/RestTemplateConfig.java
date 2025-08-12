package com.incheonai.chatbotbackend.config;

import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.BufferingClientHttpRequestFactory;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.web.client.RestTemplate;

import java.time.Duration;

@Configuration
public class RestTemplateConfig {

    @Bean
    public RestTemplate restTemplate(RestTemplateBuilder builder) {
        // SimpleClientHttpRequestFactory를 생성하여 타임아웃 설정을 적용합니다.
        SimpleClientHttpRequestFactory requestFactory = new SimpleClientHttpRequestFactory();
        requestFactory.setConnectTimeout(Duration.ofSeconds(5)); // 연결 타임아웃 5초
        requestFactory.setReadTimeout(Duration.ofSeconds(30));   // 읽기 타임아웃 30초

        return builder
                // BufferingClientHttpRequestFactory를 사용하여 chunked 전송을 비활성화하고,
                // 위에서 생성한 requestFactory를 감싸서 타임아웃 설정을 적용합니다.
                .requestFactory(() -> new BufferingClientHttpRequestFactory(requestFactory))
                .build();
    }
}
