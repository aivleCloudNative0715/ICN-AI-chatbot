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
import org.springframework.security.oauth2.core.OAuth2Error;
import org.springframework.security.oauth2.core.user.DefaultOAuth2User;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
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

    private User saveOrUpdate(OAuthAttributes attributes) {
        // [수정] 탈퇴한 회원을 포함하여 가장 최근 기록을 조회합니다.
        Optional<User> lastUserOptional = userRepository.findLastByGoogleIdWithDeleted(attributes.getGoogleId());

        if (lastUserOptional.isPresent()) {
            User lastUser = lastUserOptional.get();
            // 1. 탈퇴한 계정인지 확인
            if (lastUser.getDeletedAt() != null) {
                long daysSinceDeletion = ChronoUnit.DAYS.between(lastUser.getDeletedAt(), LocalDateTime.now());
                // 2. 탈퇴 후 30일이 지나지 않았다면, 에러를 발생시켜 로그인을 막습니다.
                if (daysSinceDeletion < 30) {
                    long daysLeft = 30 - daysSinceDeletion;
                    String errorMessage = "탈퇴한 계정입니다. " + daysLeft + "일 후에 다시 시도해주세요.";
                    // OAuth2 표준 에러를 사용하여 실패 핸들러로 정보를 전달합니다.
                    OAuth2Error error = new OAuth2Error("deleted_account", errorMessage, "");
                    throw new OAuth2AuthenticationException(error, error.toString());
                }
                // 30일이 지났다면, 재가입으로 간주하고 아래의 신규 가입 로직을 따릅니다.
            } else {
                // 3. 활성 계정이면, 정상적으로 로그인 처리합니다.
                lastUser.updateLastLogin();
                return userRepository.save(lastUser);
            }
        }

        // 4. 로컬 계정으로 이미 가입된 이메일인지 확인
        Optional<User> userByEmailOptional = userRepository.findByUserId(attributes.getEmail());
        if (userByEmailOptional.isPresent()) {
            throw new BusinessException(HttpStatus.CONFLICT, "이미 가입된 이메일입니다. 아이디와 비밀번호로 로그인해주세요.");
        }

        // 5. 완전 신규 가입 또는 30일이 지난 후 재가입 처리
        User newUser = attributes.toEntity();
        newUser.updateLastLogin();
        return userRepository.save(newUser);
    }
}