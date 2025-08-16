package com.incheonai.chatbotbackend.service;

import com.incheonai.chatbotbackend.domain.jpa.*;
import com.incheonai.chatbotbackend.dto.*;
import com.incheonai.chatbotbackend.repository.jpa.InquiryRepository;
import com.incheonai.chatbotbackend.repository.jpa.specification.InquirySpecification; // Specification 임포트
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.domain.Specification; // Specification 임포트
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.util.List;

import com.incheonai.chatbotbackend.domain.jpa.BoardCategory;
import com.incheonai.chatbotbackend.domain.jpa.Urgency;
import com.incheonai.chatbotbackend.dto.InquiryCountsDto;

@Service
@RequiredArgsConstructor
public class AdminInquiryService {
    private final InquiryRepository inquiryRepository;

    /** 문의 목록 동적 조회 */
    @Transactional(readOnly = true)
    public Page<InquiryDto> getInquiries(
            InquiryStatus status,
            List<Urgency> urgencies,
            BoardCategory category,
            String search,
            LocalDate startDate,
            LocalDate endDate,
            Pageable pageable) {

        // 1. Specification을 조합하기 위한 변수를 선언합니다.
        Specification<Inquiry> spec = (root, query, criteriaBuilder) -> criteriaBuilder.conjunction();

        // 2. 각 파라미터가 존재하면, .and()를 이용해 조건을 추가합니다.
        if (status != null) {
            spec = spec.and(InquirySpecification.hasStatus(status));
        }
        if (urgencies != null && !urgencies.isEmpty()) {
            spec = spec.and(InquirySpecification.hasUrgenciesIn(urgencies));
        }
        if (category != null) {
            spec = spec.and(InquirySpecification.hasCategory(category));
        }
        if (search != null && !search.isBlank()) {
            spec = spec.and(InquirySpecification.containsTextInTitleAndContent(search));
        }
        if (startDate != null || endDate != null) {
            LocalDateTime startDateTime = (startDate != null) ? startDate.atStartOfDay() : null;
            LocalDateTime endDateTime = (endDate != null) ? endDate.atTime(LocalTime.MAX) : null;
            spec = spec.and(InquirySpecification.isCreatedAtBetween(startDateTime, endDateTime));
        }

        return inquiryRepository.findAll(spec, pageable).map(InquiryDto::fromEntity);
    }

    /** 문의 건수 조회 (전체 기간 기준) */
    @Transactional(readOnly = true)
    public InquiryCountsDto getInquiryCounts() {
        long total = inquiryRepository.count();
        long pending = inquiryRepository.countByStatus(InquiryStatus.PENDING);
        long resolved = inquiryRepository.countByStatus(InquiryStatus.RESOLVED);
        long inquiryCount = inquiryRepository.countByCategory(BoardCategory.INQUIRY);
        long suggestionCount = inquiryRepository.countByCategory(BoardCategory.SUGGESTION);
        long highCount = inquiryRepository.countByUrgency(Urgency.HIGH);
        long mediumCount = inquiryRepository.countByUrgency(Urgency.MEDIUM);
        long lowCount = inquiryRepository.countByUrgency(Urgency.LOW);

        return InquiryCountsDto.builder()
                .total(total)
                .pending(pending)
                .resolved(resolved)
                .inquiry(inquiryCount)
                .suggestion(suggestionCount)
                .high(highCount)
                .medium(mediumCount)
                .low(lowCount)
                .build();
    }

    /** 단일 문의 상세 조회 */
    @Transactional(readOnly = true)
    public InquiryDetailDto getInquiryDetail(Integer inquiryId) {
        return inquiryRepository.findById(inquiryId)
                .map(InquiryDetailDto::fromEntity)
                .orElseThrow(() -> new IllegalArgumentException("존재하지 않는 문의입니다."));
    }

    /** 문의 긴급도(Urgency) 수정 */
    @Transactional
    public void updateUrgency(Integer inquiryId, Urgency urgency) {
        Inquiry inquiry = inquiryRepository.findById(inquiryId)
                .orElseThrow(() -> new IllegalArgumentException("문의를 찾을 수 없습니다."));
        inquiry.setUrgency(urgency);
    }

    /** 문의 답변 등록 및 수정 */
    @Transactional
    public InquiryDetailDto processAnswer(Integer inquiryId, InquiryAnswerRequestDto request) {
        Inquiry inquiry = inquiryRepository.findById(inquiryId)
                .orElseThrow(() -> new IllegalArgumentException("답변할 문의를 찾을 수 없습니다."));

        inquiry.setAnswer(request.content());
        inquiry.setAdminId(request.adminId());
        inquiry.setStatus(InquiryStatus.RESOLVED);

        // 2. 요청에 urgency 값이 있으면 함께 업데이트합니다.
        if (request.urgency() != null) {
            inquiry.setUrgency(request.urgency());
        }

        return InquiryDetailDto.fromEntity(inquiry);
    }
}
