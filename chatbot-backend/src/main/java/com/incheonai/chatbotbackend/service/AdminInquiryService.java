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
import org.springframework.util.StringUtils;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.incheonai.chatbotbackend.domain.jpa.BoardCategory;
import com.incheonai.chatbotbackend.domain.jpa.Urgency;
import com.incheonai.chatbotbackend.dto.InquiryCountsDto;

@Service
@RequiredArgsConstructor
public class AdminInquiryService {
    private final InquiryRepository inquiryRepository;

    /** 문의 목록 동적 조회 (검색 기능 포함) */
    @Transactional(readOnly = true)
    public Page<InquiryDto> getInquiries(String status, Urgency urgency, BoardCategory category, String search, Pageable pageable) {

        // 1. 적용할 Specification 조건을 담을 리스트를 생성합니다.
        List<Specification<Inquiry>> specs = new ArrayList<>();

        // 2. 각 파라미터가 존재할 경우에만 리스트에 해당 Specification을 추가합니다.
        if (StringUtils.hasText(status)) {
            specs.add(InquirySpecification.hasStatus(InquiryStatus.valueOf(status.toUpperCase())));
        }
        if (urgency != null) {
            specs.add(InquirySpecification.hasUrgency(urgency));
        }
        if (category != null) {
            specs.add(InquirySpecification.hasCategory(category));
        }
        if (StringUtils.hasText(search)) {
            specs.add(InquirySpecification.containsText(search));
        }

        // 3. 리스트에 있는 모든 Specification을 .and() 연산으로 조합합니다.
        //    리스트가 비어있으면 null이 되어, 전체 조회를 수행합니다.
        Specification<Inquiry> finalSpec = specs.stream()
                .reduce(Specification::and)
                .orElse(null);

        // 4. 최종 조합된 Specification으로 데이터를 조회합니다.
        return inquiryRepository.findAll(finalSpec, pageable).map(InquiryDto::fromEntity);
    }

    /** 문의 건수 조회 */
    @Transactional(readOnly = true)
    public InquiryCountsDto getInquiryCounts(LocalDateTime start, LocalDateTime end) {
        long total = inquiryRepository.countByCreatedAtBetween(start, end);
        long pending = inquiryRepository.countByStatusAndCreatedAtBetween(InquiryStatus.PENDING, start, end);
        long resolved = total - pending; // 전체 - 미처리 = 완료

        // 카테고리별 건수 조회
        long inquiryCount = inquiryRepository.countByCategoryAndCreatedAtBetween(BoardCategory.INQUIRY, start, end);
        long suggestionCount = inquiryRepository.countByCategoryAndCreatedAtBetween(BoardCategory.SUGGESTION, start, end);

        // 중요도별 건수 조회
        long highCount = inquiryRepository.countByUrgencyAndCreatedAtBetween(Urgency.HIGH, start, end);
        long mediumCount = inquiryRepository.countByUrgencyAndCreatedAtBetween(Urgency.MEDIUM, start, end);
        long lowCount = inquiryRepository.countByUrgencyAndCreatedAtBetween(Urgency.LOW, start, end);

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

    /** 문의 상태(Status) 수정 */
    @Transactional
    public void updateStatus(Integer inquiryId, InquiryStatus status) {
        Inquiry inquiry = inquiryRepository.findById(inquiryId)
                .orElseThrow(() -> new IllegalArgumentException("문의를 찾을 수 없습니다."));
        inquiry.setStatus(status);
    }

    /** 문의 답변 등록 및 수정 */
    @Transactional
    public InquiryDetailDto processAnswer(Integer inquiryId, InquiryAnswerRequestDto request) {
        Inquiry inquiry = inquiryRepository.findById(inquiryId)
                .orElseThrow(() -> new IllegalArgumentException("답변할 문의를 찾을 수 없습니다."));

        inquiry.setAnswer(request.content());
        inquiry.setAdminId(request.adminId());
        // 답변 등록 시 상태를 자동으로 '답변완료'로 변경
        inquiry.setStatus(InquiryStatus.RESOLVED);

        return InquiryDetailDto.fromEntity(inquiry);
    }

    /** 문의 삭제 */
    @Transactional
    public void deleteInquiry(Integer inquiryId) {
        inquiryRepository.deleteById(inquiryId);
    }
}
