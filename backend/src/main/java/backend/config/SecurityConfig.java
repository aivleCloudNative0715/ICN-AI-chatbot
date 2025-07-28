package backend.config;

import backend.config.JwtUtil;
import backend.service.TokenBlacklist;
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
    private final TokenBlacklist blacklist;

    public SecurityConfig(JwtUtil jwtUtil, TokenBlacklist blacklist) {
        this.jwtUtil = jwtUtil;
        this.blacklist = blacklist;
    }

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
                // CSRF 비활성화
                .csrf(csrf -> csrf.disable())
                // 세션을 Stateless로
                .sessionManagement(sm ->
                        sm.sessionCreationPolicy(SessionCreationPolicy.STATELESS)
                )
                // 경로별 권한 설정
                .authorizeHttpRequests(authz -> authz
                        // 1) OpenAPI JSON/YAML 스펙 (루트 + 서브경로) 모두 공개
                        .requestMatchers(
                                "/v3/api-docs",          // 루트 스펙
                                "/v3/api-docs/**",       // 서브 경로
                                "/v3/api-docs.yaml"      // YAML 스펙
                        ).permitAll()

                        // 2) Swagger UI 정적 리소스 공개
                        .requestMatchers(
                                "/swagger-ui.html",
                                "/swagger-ui/index.html",
                                "/swagger-ui/**"
                        ).permitAll()

                        // 3) 스프링 기본 오류 페이지
                        .requestMatchers("/error").permitAll()

                        // 4) 회원가입·로그인
                        .requestMatchers(HttpMethod.POST,
                                "/api/auth/register",
                                "/api/auth/login",
                                "/api/auth/login-admin"
                        ).permitAll()

                        // 5) 로그아웃(인증된 사용자)
                        .requestMatchers(HttpMethod.POST, "/api/auth/logout").authenticated()

                        // 6) 공개 GET
                        .requestMatchers(HttpMethod.GET,
                                "/api/chats",
                                "/"
                        ).permitAll()

                        // 7) ADMIN 전용
                        .requestMatchers(HttpMethod.GET, "/api/auth/users").hasRole("ADMIN")
                        .requestMatchers("/admin/**").hasRole("ADMIN")

                        // 8) 그 외 요청은 모두 인증 필요
                        .anyRequest().authenticated()
                )
                // JWT + 블랙리스트 필터 등록
                .addFilterBefore(
                        new JwtFilter(jwtUtil, blacklist),
                        UsernamePasswordAuthenticationFilter.class
                );

        return http.build();
    }

    /**
     * JWT 토큰 검사 + 블랙리스트 체크 필터
     */
    private static class JwtFilter extends OncePerRequestFilter {
        private final JwtUtil jwtUtil;
        private final TokenBlacklist blacklist;

        public JwtFilter(JwtUtil jwtUtil, TokenBlacklist blacklist) {
            this.jwtUtil = jwtUtil;
            this.blacklist = blacklist;
        }

        @Override
        protected void doFilterInternal(
                HttpServletRequest req,
                HttpServletResponse res,
                FilterChain chain
        ) throws ServletException, IOException {
            String header = req.getHeader("Authorization");
            if (header != null && header.startsWith("Bearer ")) {
                String token = header.substring(7);
                if (!blacklist.contains(token)) {
                    try {
                        Claims claims = jwtUtil.parseToken(token);
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
            }
            chain.doFilter(req, res);
        }
    }

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }
}
