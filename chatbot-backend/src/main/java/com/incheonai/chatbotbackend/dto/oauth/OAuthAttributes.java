package com.incheonai.chatbotbackend.dto.oauth;

import com.incheonai.chatbotbackend.domain.jpa.LoginProvider;
import com.incheonai.chatbotbackend.domain.jpa.User;
import lombok.Builder;
import lombok.Getter;

import java.util.Map;

/**
 * OAuth2 제공자(Google)로부터 받은 사용자 정보를 담는 DTO
 */
@Getter
public class OAuthAttributes {
    private Map<String, Object> attributes;
    private String nameAttributeKey;
    private String name;
    private String email;
    private String googleId;

    @Builder
    public OAuthAttributes(Map<String, Object> attributes, String nameAttributeKey, String name, String email, String googleId) {
        this.attributes = attributes;
        this.nameAttributeKey = nameAttributeKey;
        this.name = name;
        this.email = email;
        this.googleId = googleId;
    }

    public static OAuthAttributes of(String registrationId, String userNameAttributeName, Map<String, Object> attributes) {
        // 현재는 google만 처리
        return ofGoogle(userNameAttributeName, attributes);
    }

    private static OAuthAttributes ofGoogle(String userNameAttributeName, Map<String, Object> attributes) {
        return OAuthAttributes.builder()
                .name((String) attributes.get("name"))
                .email((String) attributes.get("email"))
                .googleId((String) attributes.get("sub")) // 구글의 고유 식별자
                .attributes(attributes)
                .nameAttributeKey(userNameAttributeName)
                .build();
    }

    // User 엔티티 생성 (회원가입 시 사용)
    public User toEntity() {
        return User.builder()
                .userId(email) // 초기 userId는 이메일로 설정
                .googleId(googleId)
                .loginProvider(LoginProvider.GOOGLE)
                .build();
    }
}