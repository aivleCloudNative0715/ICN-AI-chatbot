package com.incheonai.chatbotbackend.controller;

import com.incheonai.chatbotbackend.dto.CheckIdRequestDto;
import com.incheonai.chatbotbackend.dto.CheckIdResponseDto;
import com.incheonai.chatbotbackend.dto.LoginRequestDto;
import com.incheonai.chatbotbackend.service.AuthService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/auth")
public class AuthController {

    private final AuthService authService;

    @PostMapping("/check-id")
    public ResponseEntity<CheckIdResponseDto> checkId(@Valid @RequestBody CheckIdRequestDto requestDto) {
        boolean isAvailable = authService.checkIdDuplication(requestDto.userId());
        return ResponseEntity.ok(new CheckIdResponseDto(isAvailable));
    }

    @PostMapping("/login")
    public ResponseEntity<Object> login(@Valid @RequestBody LoginRequestDto requestDto) {
        Object loginResponse = authService.login(requestDto);
        return ResponseEntity.ok(loginResponse);
    }
}
