package com.incheonai.chatbotbackend.service;

import com.incheonai.chatbotbackend.domain.jpa.Inquiry;
import com.incheonai.chatbotbackend.dto.InquiryDetailDto;
import com.incheonai.chatbotbackend.dto.InquiryDto;
import com.incheonai.chatbotbackend.dto.InquiryRequestDto;
import com.incheonai.chatbotbackend.repository.jpa.InquiryRepository;
import com.incheonai.chatbotbackend.repository.jpa.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class InquiryService {

    private final InquiryRepository inquiryRepository;
    private final UserRepository userRepository;

    /** 문의 등록 (POST /inquiries) */
    @Transactional
    public InquiryDto createInquiry(String userId, InquiryRequestDto request) {
        // 사용자 존재 여부 확인 (사용 안 하는 변수 경고 제거용으로 existsById 사용)
        if (!userRepository.existsById(userId)) {
            throw new IllegalArgumentException("존재하지 않는 사용자입니다.");
        }

        Inquiry inquiry = Inquiry.builder()
                .userId(userId)
                .title(request.title())
                .content(request.content())
                .category(request.category())
                .urgency(request.urgency())
                .build();

        Inquiry saved = inquiryRepository.save(inquiry);
        return InquiryDto.fromEntity(saved);
    }

    /** 내 문의 목록 조회 (GET /users/me/inquiries?page=&size=&status=) */
    @Transactional(readOnly = true)
    public Page<InquiryDto> getMyInquiries(String userId, String status, Pageable pageable) {
        String effectiveStatus = (status == null) ? "" : status;
        return inquiryRepository
                .findByUserIdAndStatusContaining(userId, effectiveStatus, pageable)
                .map(InquiryDto::fromEntity);
    }

    /** 단일 문의 상세 조회 (GET /users/me/inquiries/{inquiry_id}) */
    @Transactional(readOnly = true)
    public InquiryDetailDto getMyInquiryDetail(String userId, Integer inquiryId) {
        Inquiry inquiry = inquiryRepository
                .findByInquiryIdAndUserId(inquiryId, userId)
                .orElseThrow(() -> new IllegalArgumentException("문의 정보를 찾을 수 없습니다."));
        return InquiryDetailDto.fromEntity(inquiry);
    }

    /** 내 문의 수정 (PUT /users/me/inquiries/{inquiry_id}) */
    @Transactional
    public InquiryDto updateMyInquiry(String userId, Integer inquiryId, InquiryRequestDto request) {
        Inquiry inquiry = inquiryRepository
                .findByInquiryIdAndUserId(inquiryId, userId)
                .orElseThrow(() -> new IllegalArgumentException("문의 정보를 찾을 수 없습니다."));

        inquiry.setTitle(request.title());
        inquiry.setContent(request.content());
        inquiry.setCategory(request.category());
        inquiry.setUrgency(request.urgency());

        return InquiryDto.fromEntity(inquiryRepository.save(inquiry));
    }

    /** 내 문의 삭제 (DELETE /users/me/inquiries/{inquiry_id}) */
    @Transactional
    public void deleteMyInquiry(String userId, Integer inquiryId) {
        Inquiry inquiry = inquiryRepository
                .findByInquiryIdAndUserId(inquiryId, userId)
                .orElseThrow(() -> new IllegalArgumentException("문의 정보를 찾을 수 없습니다."));
        inquiryRepository.delete(inquiry);
    }
}
