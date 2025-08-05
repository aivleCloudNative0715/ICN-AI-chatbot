package com.incheonai.chatbotbackend.service;

import com.incheonai.chatbotbackend.domain.jpa.User;
import com.incheonai.chatbotbackend.dto.oauth.OAuthAttributes;
import com.incheonai.chatbotbackend.exception.BusinessException;
import com.incheonai.chatbotbackend.repository.jpa.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.oauth2.client.userinfo.DefaultOAuth2UserService;
import org.springframework.security.oauth2.client.userinfo.OAuth2UserRequest;
import org.springframework.security.oauth2.client.userinfo.OAuth2UserService;
import org.springframework.security.oauth2.core.OAuth2AuthenticationException;
import org.springframework.security.oauth2.core.user.DefaultOAuth2User;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.Collections;
import java.util.Optional;

@Service
@RequiredArgsConstructor
public class CustomOAuth2UserService implements OAuth2UserService<OAuth2UserRequest, OAuth2User> {

    private final UserRepository userRepository;

    @Override
    @Transactional
    public OAuth2User loadUser(OAuth2UserRequest userRequest) throws OAuth2AuthenticationException {
        OAuth2UserService<OAuth2UserRequest, OAuth2User> delegate = new DefaultOAuth2UserService();
        OAuth2User oAuth2User = delegate.loadUser(userRequest);

        String registrationId = userRequest.getClientRegistration().getRegistrationId();
        String userNameAttributeName = userRequest.getClientRegistration().getProviderDetails().getUserInfoEndpoint().getUserNameAttributeName();

        OAuthAttributes attributes = OAuthAttributes.of(registrationId, userNameAttributeName, oAuth2User.getAttributes());

        User user = saveOrUpdate(attributes);

        return new DefaultOAuth2User(
                Collections.singleton(new SimpleGrantedAuthority("ROLE_USER")),
                attributes.getAttributes(),
                attributes.getNameAttributeKey()
        );
    }

    /**
     * OAuth 사용자를 저장하거나 업데이트합니다.
     * 로컬 계정과의 연동은 허용하지 않습니다.
     */
    private User saveOrUpdate(OAuthAttributes attributes) {
        // 1. Google ID로 사용자 조회
        Optional<User> userOptional = userRepository.findByGoogleId(attributes.getGoogleId());

        if (userOptional.isPresent()) {
            // 이미 구글로 가입된 사용자. 마지막 로그인 시간 업데이트 후 반환
            User user = userOptional.get();
            user.updateLastLogin();
            return userRepository.save(user);
        }

        // 2. 계정 연동 방지: 같은 이메일(userId)로 가입된 계정이 있는지 확인
        // OAuthAttributes에서 toEntity()는 email을 userId로 사용합니다.
        Optional<User> userByEmailOptional = userRepository.findByUserId(attributes.getEmail());
        if (userByEmailOptional.isPresent()) {
            // 이미 해당 이메일을 아이디로 사용하는 계정이 존재하면 에러 발생
            throw new BusinessException(HttpStatus.CONFLICT, "이미 가입된 이메일입니다. 아이디와 비밀번호로 로그인해주세요.");
        }

        // 3. 완전 신규 사용자인 경우, 새로 생성
        User user = attributes.toEntity();
        user.updateLastLogin(); // 최초 로그인 시간 기록
        return userRepository.save(user);
    }
}