package com.incheonai.chatbotbackend.exception;

import com.incheonai.chatbotbackend.dto.ErrorResponse;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice // 모든 @RestController에서 발생하는 예외를 잡아줍니다.
public class GlobalExceptionHandler {

    @ExceptionHandler(BusinessException.class) // BusinessException을 처리하도록 변경
    public ResponseEntity<ErrorResponse> handleBusinessException(BusinessException ex) {
        // 예외 객체에서 직접 HttpStatus와 메시지를 가져옵니다.
        ErrorResponse response = ErrorResponse.of(ex.getHttpStatus(), ex.getMessage());
        return new ResponseEntity<>(response, ex.getHttpStatus());
    }

    /**
     * 위에서 처리하지 못한 모든 예외를 처리합니다.
     */
    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleException(Exception ex) {
        // 예상치 못한 모든 에러는 500 Internal Server Error로 처리합니다.
        ErrorResponse response = ErrorResponse.of(HttpStatus.INTERNAL_SERVER_ERROR, ex.getMessage());
        return new ResponseEntity<>(response, HttpStatus.INTERNAL_SERVER_ERROR);
    }
}