package backend.config;

import backend.config.JwtUtil;
import io.jsonwebtoken.Claims;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpMethod;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.util.Collections;

@Configuration
public class SecurityConfig {

    private final JwtUtil jwtUtil;

    public SecurityConfig(JwtUtil jwtUtil) {
        this.jwtUtil = jwtUtil;
    }

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
                // CSRF 비활성화
                .csrf(csrf -> csrf.disable())
                // 세션 정책을 Stateless로 설정
                .sessionManagement(sm -> sm.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
                // 경로별 권한 설정
                .authorizeHttpRequests(authz -> authz
                        // 에러 핸들러 경로 공개
                        .requestMatchers("/error").permitAll()
                        // 인증 없이 허용할 POST 엔드포인트
                        .requestMatchers(HttpMethod.POST,
                                "/api/auth/register",
                                "/api/auth/login",
                                "/api/auth/login-admin"
                        ).permitAll()
                        // 인증 없이 허용할 GET 엔드포인트
                        .requestMatchers(HttpMethod.GET,
                                "/api/chats",
                                "/"
                        ).permitAll()
                        // ADMIN 전용: 사용자 리스트 조회 및 /admin/**
                        .requestMatchers(HttpMethod.GET, "/api/auth/users").hasRole("ADMIN")
                        .requestMatchers("/admin/**").hasRole("ADMIN")
                        // 그 외 요청은 인증 필요
                        .anyRequest().authenticated()
                )
                // JWT 검증 필터 등록
                .addFilterBefore(new JwtFilter(jwtUtil),
                        UsernamePasswordAuthenticationFilter.class
                );

        return http.build();
    }

    /** JWT 토큰 검사 필터 */
    private static class JwtFilter extends OncePerRequestFilter {
        private final JwtUtil jwtUtil;
        public JwtFilter(JwtUtil jwtUtil) { this.jwtUtil = jwtUtil; }

        @Override
        protected void doFilterInternal(
                HttpServletRequest req,
                HttpServletResponse res,
                FilterChain chain
        ) throws ServletException, IOException {
            String header = req.getHeader("Authorization");
            if (header != null && header.startsWith("Bearer ")) {
                try {
                    Claims claims = jwtUtil.parseToken(header.substring(7));
                    var auth = new UsernamePasswordAuthenticationToken(
                            claims.getSubject(),
                            null,
                            Collections.singletonList(
                                    new SimpleGrantedAuthority("ROLE_" + claims.get("role", String.class))
                            )
                    );
                    SecurityContextHolder.getContext().setAuthentication(auth);
                } catch (Exception ignored) {}
            }
            chain.doFilter(req, res);
        }
    }

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }
}
