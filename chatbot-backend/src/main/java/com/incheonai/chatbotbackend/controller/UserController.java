package com.incheonai.chatbotbackend.controller;

import com.incheonai.chatbotbackend.dto.LoginRequestDto;
import com.incheonai.chatbotbackend.dto.LoginResponseDto;
import com.incheonai.chatbotbackend.dto.SignUpRequestDto;
import com.incheonai.chatbotbackend.service.UserService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;
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

    /**
     * 현재 로그인된 사용자 정보 조회 엔드포인트
     * @param authentication (JwtAuthenticationFilter에서 SecurityContext에 저장한 인증 정보)
     * @return ResponseEntity<LoginResponseDto>
     */
    @GetMapping("/me")
    public ResponseEntity<LoginResponseDto> getMyInfo(Authentication authentication) {
        // Authentication 객체의 getName()은 토큰 생성 시 넣었던 subject(userId)를 반환합니다.
        String userId = authentication.getName();
        LoginResponseDto userInfo = userService.getUserInfo(userId);
        return ResponseEntity.ok(userInfo);
    }

    /**
     * 현재 로그인된 사용자 계정을 삭제합니다.
     * @param authentication (현재 인증 정보)
     * @param request (현재 요청 객체, 토큰을 추출하기 위해 사용)
     * @return ResponseEntity<String>
     */
    @DeleteMapping("/me")
    public ResponseEntity<String> deleteMyAccount(Authentication authentication, HttpServletRequest request) {
        // 1. 현재 사용자의 ID를 가져옵니다.
        String userId = authentication.getName();

        // 2. 현재 사용 중인 토큰을 가져옵니다.
        String token = resolveToken(request);

        // 3. 사용자 계정을 soft-delete 합니다.
        userService.deleteAccount(userId);

        // 4. 현재 사용 중인 토큰을 무효화하여 즉시 로그아웃 처리합니다.
        if (token != null) {
            authService.logout(token);
        }

        return ResponseEntity.ok("계정이 성공적으로 삭제되었습니다.");
    }

    private String resolveToken(HttpServletRequest request) {
        String bearerToken = request.getHeader("Authorization");
        if (StringUtils.hasText(bearerToken) && bearerToken.startsWith("Bearer ")) {
            return bearerToken.substring(7);
        }
        return null;
    }
}