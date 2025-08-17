// com/incheonai/chatbotbackend/dto/external/ApiResult.java
package com.incheonai.chatbotbackend.dto.external;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@JsonIgnoreProperties(ignoreUnknown = true)
public class ApiResult<T> {
    // 실제 JSON의 "response" 키와 매칭되는 필드
    private ApiResponseData<T> response;
}