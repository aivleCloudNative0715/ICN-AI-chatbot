package com.incheonai.chatbotbackend.config;

import com.incheonai.chatbotbackend.config.jwt.JwtAuthenticationFilter;
import com.incheonai.chatbotbackend.config.jwt.JwtTokenProvider;
import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;

@Configuration
@EnableWebSecurity
@RequiredArgsConstructor
public class SecurityConfig {

    private final JwtTokenProvider jwtTokenProvider;
    private final RedisTemplate<String, Object> redisTemplate;

    // Swagger UI 접속을 위한 경로들
    private static final String[] SWAGGER_URL_PATTERNS = {
            "/v3/api-docs/**", "/swagger-ui/**", "/swagger-ui.html"
    };

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
                .csrf(csrf -> csrf.disable())
                .sessionManagement(session -> session.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
                .authorizeHttpRequests(authz -> authz
                        // 회원가입 및 로그인 API 경로는 인증 없이 허용
                        .requestMatchers("/api/auth/**", "/api/users/signup").permitAll()
                        // Swagger UI 관련 경로들은 인증 없이 허용
                        .requestMatchers(SWAGGER_URL_PATTERNS).permitAll()
                        // 그 외 모든 요청은 인증 필요
                        .anyRequest().authenticated()
                )
                // JwtAuthenticationFilter를 UsernamePasswordAuthenticationFilter 전에 추가
                .addFilterBefore(new JwtAuthenticationFilter(jwtTokenProvider, redisTemplate), UsernamePasswordAuthenticationFilter.class);

        return http.build();
    }
}
