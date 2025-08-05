package com.incheonai.chatbotbackend.service;

import com.incheonai.chatbotbackend.repository.jpa.AdminRepository;
import com.incheonai.chatbotbackend.repository.jpa.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

import java.util.Collections;

@Service
@RequiredArgsConstructor
public class CustomUserDetailsService implements UserDetailsService {

    private final UserRepository userRepository;
    private final AdminRepository adminRepository;

    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        // User 테이블에서 먼저 찾고, 없으면 Admin 테이블에서 찾음
        return userRepository.findByUserId(username)
                .map(this::createUserDetails)
                .orElseGet(() -> adminRepository.findByAdminId(username)
                        .map(this::createAdminDetails)
                        .orElseThrow(() -> new UsernameNotFoundException(username + " -> 데이터베이스에서 찾을 수 없습니다.")));
    }

    // User 엔티티를 UserDetails 객체로 변환
    private UserDetails createUserDetails(com.incheonai.chatbotbackend.domain.jpa.User user) {
        // "ROLE_USER" 문자열을 SimpleGrantedAuthority 객체로 변환
        SimpleGrantedAuthority authority = new SimpleGrantedAuthority("ROLE_USER");

        return User.builder()
                .username(user.getUserId())
                .password(user.getPassword() != null ? user.getPassword() : "")
                .authorities(Collections.singletonList(authority)) // 수정된 부분
                .build();
    }

    // Admin 엔티티를 UserDetails 객체로 변환
    private UserDetails createAdminDetails(com.incheonai.chatbotbackend.domain.jpa.Admin admin) {
        // "ROLE_" + Enum 이름을 조합하여 SimpleGrantedAuthority 객체로 변환
        SimpleGrantedAuthority authority = new SimpleGrantedAuthority("ROLE_" + admin.getRole().name());

        return User.builder()
                .username(admin.getAdminId())
                .password(admin.getPassword())
                .authorities(Collections.singletonList(authority)) // 수정된 부분
                .build();
    }
}
