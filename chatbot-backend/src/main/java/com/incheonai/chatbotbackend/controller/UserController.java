package com.incheonai.chatbotbackend.controller;

import com.incheonai.chatbotbackend.dto.LoginRequestDto;
import com.incheonai.chatbotbackend.dto.LoginResponseDto;
import com.incheonai.chatbotbackend.dto.SignUpRequestDto;
import com.incheonai.chatbotbackend.service.UserService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import com.incheonai.chatbotbackend.service.AuthService;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.util.StringUtils;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/users")
public class UserController {

    private final UserService userService;
    private final AuthService authService;

    @PostMapping("/signup")
    public ResponseEntity<LoginResponseDto> signup(@Valid @RequestBody SignUpRequestDto requestDto) {
        // 서비스에서 토큰이 담긴 DTO를 직접 반환받음
        LoginResponseDto response = userService.signup(requestDto);
        // 상태 코드 201 Created와 함께 응답 바디에 토큰을 담아 반환
        return ResponseEntity.status(HttpStatus.CREATED).body(response);
    }

    @PostMapping("/logout")
    public ResponseEntity<String> logout(HttpServletRequest request) {
        String token = resolveToken(request);
        if (token == null) {
            return ResponseEntity.badRequest().body("토큰이 없습니다.");
        }
        authService.logout(token);
        return ResponseEntity.ok("로그아웃 되었습니다.");
    }

    private String resolveToken(HttpServletRequest request) {
        String bearerToken = request.getHeader("Authorization");
        if (StringUtils.hasText(bearerToken) && bearerToken.startsWith("Bearer ")) {
            return bearerToken.substring(7);
        }
        return null;
    }
}