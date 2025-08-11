package com.incheonai.chatbotbackend.service;

import com.incheonai.chatbotbackend.domain.jpa.Inquiry;
import com.incheonai.chatbotbackend.domain.jpa.InquiryStatus;
import com.incheonai.chatbotbackend.dto.InquiryDetailDto;
import com.incheonai.chatbotbackend.dto.InquiryDto;
import com.incheonai.chatbotbackend.dto.InquiryRequestDto;
import com.incheonai.chatbotbackend.repository.jpa.InquiryRepository;
import com.incheonai.chatbotbackend.repository.jpa.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.dao.InvalidDataAccessApiUsageException;
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
        // PK 기반 existsById(userId) 아님! user_id 컬럼 기반으로 체크
        if (!userRepository.existsByUserId(userId)) {
            throw new IllegalArgumentException("존재하지 않는 사용자입니다.");
        }

        Inquiry inquiry = Inquiry.builder()
                .userId(userId)
                .title(request.title())
                .content(request.content())
                .category(request.category())
                .urgency(request.urgency())
                // 상태 기본값 필요하면 주석 해제
                // .status(InquiryStatus.PENDING)
                .build();

        return InquiryDto.fromEntity(inquiryRepository.save(inquiry));
    }

    /** 내 문의 목록 조회 (GET /users/me/inquiries?page=&size=&status=) */
    @Transactional(readOnly = true)
    public Page<InquiryDto> getMyInquiries(String userId, String status, Pageable pageable) {
        Page<Inquiry> page;
        if (status == null || status.isBlank()) {
            page = inquiryRepository.findByUserId(userId, pageable);
        } else {
            try {
                InquiryStatus st = InquiryStatus.valueOf(status);
                page = inquiryRepository.findByUserIdAndStatus(userId, status, pageable);
            } catch (IllegalArgumentException ex) {
                // 잘못된 status 문자열이면 빈 페이지 반환(또는 커스텀 예외로 400 처리)
                throw new InvalidDataAccessApiUsageException("유효하지 않은 상태 값입니다. (예: PENDING, RESOLVED)");
            }
        }
        return page.map(InquiryDto::fromEntity);
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
