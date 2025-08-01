package com.incheonai.chatbotbackend.dto;

import lombok.Builder;
import lombok.Getter;
import org.springframework.http.HttpStatus;

import java.time.LocalDateTime;

@Getter
@Builder
public class ErrorResponse {

    private final LocalDateTime timestamp = LocalDateTime.now();
    private final int status;
    private final String error;
    private final String message;

    public static ErrorResponse of(HttpStatus httpStatus, String message) {
        return ErrorResponse.builder()
                .status(httpStatus.value())
                .error(httpStatus.getReasonPhrase())
                .message(message)
                .build();
    }
}