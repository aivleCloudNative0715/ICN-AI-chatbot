package com.incheonai.chatbotbackend.service;

import com.incheonai.chatbotbackend.domain.jpa.Inquiry;
import com.incheonai.chatbotbackend.domain.jpa.InquiryAnswer;
import com.incheonai.chatbotbackend.domain.jpa.InquiryStatus;
import com.incheonai.chatbotbackend.dto.InquiryAnswerRequestDto;
import com.incheonai.chatbotbackend.dto.InquiryAnswerResponseDto;
import com.incheonai.chatbotbackend.dto.InquiryDetailDto;
import com.incheonai.chatbotbackend.dto.InquiryDto;
import com.incheonai.chatbotbackend.repository.jpa.InquiryAnswerRepository;
import com.incheonai.chatbotbackend.repository.jpa.InquiryRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.Map;

@Service
@RequiredArgsConstructor
public class InquiryService {
    private final InquiryRepository inquiryRepository;
    private final InquiryAnswerRepository answerRepository;

    @Transactional(readOnly = true)
    public Page<InquiryDto> getInquiries(String status, Integer urgency, String category, String search,
                                         Pageable pageable) {
        return inquiryRepository.findAll(pageable)
                .map(InquiryDto::fromEntity);
    }

    @Transactional(readOnly = true)
    public Map<String, Long> getInquiryCounts(LocalDateTime start, LocalDateTime end) {
        long total    = inquiryRepository.countByCreatedAtBetween(start, end);
        long pending  = inquiryRepository.countByStatusAndCreatedAtBetween(InquiryStatus.PENDING, start, end);
        long resolved = inquiryRepository.countByStatusAndCreatedAtBetween(InquiryStatus.RESOLVED, start, end);
        return Map.of("total", total, "pending", pending, "resolved", resolved);
    }

    /**
     * 단일 문의 상세 조회
     * @param inquiryId 조회할 문의의 PK (문자열)
     * @return InquiryDetailDto 변환된 문의 상세 DTO
     */
    @Transactional(readOnly = true)
    public InquiryDetailDto getInquiryDetail(String inquiryId) {
        Inquiry in = inquiryRepository.findById(inquiryId)
                .orElseThrow(() -> new IllegalArgumentException("존재하지 않는 문의입니다."));
        return InquiryDetailDto.fromEntity(in);
    }

    /**
     * 문의 긴급도(Urgency) 수정
     * @param inquiryId 수정할 문의의 PK
     * @param urgency   새 긴급도 값 (예: 1, 2, 3)
     */
    @Transactional
    public void updateUrgency(String inquiryId, Integer urgency) {
        Inquiry in = inquiryRepository.findById(inquiryId)
                .orElseThrow(() -> new IllegalArgumentException("문의가 없습니다."));
        in.setUrgency(urgency);
        inquiryRepository.save(in);
    }

    /**
     * 문의 상태(Status) 수정
     * @param inquiryId 수정할 문의의 PK
     * @param status    새 상태 이름 (PENDING, RESOLVED 등)
     */
    @Transactional
    public void updateStatus(String inquiryId, String status) {
        Inquiry in = inquiryRepository.findById(inquiryId)
                .orElseThrow(() -> new IllegalArgumentException("문의가 없습니다."));
        in.setStatus(InquiryStatus.valueOf(status));
        inquiryRepository.save(in);
    }

    /** 문의 답변 등록 */
    @Transactional
    public InquiryAnswerResponseDto addAnswer(String inquiryId, InquiryAnswerRequestDto request) {
        InquiryAnswer ans = InquiryAnswer.builder()
                .inquiryId(Integer.valueOf(inquiryId))
                .adminId(request.adminId())
                .content(request.content())
                .build();
        InquiryAnswer saved = answerRepository.save(ans);
        return InquiryAnswerResponseDto.fromEntity(saved);
    }

    /** 문의 답변 수정 */
    @Transactional
    public InquiryAnswerResponseDto updateAnswer(String inquiryId, String answerId, InquiryAnswerRequestDto request) {
        InquiryAnswer ans = answerRepository.findById(Integer.valueOf(answerId))
                .orElseThrow(() -> new IllegalArgumentException("답변이 없습니다."));
        ans.setContent(request.content());
        InquiryAnswer updated = answerRepository.save(ans);
        return InquiryAnswerResponseDto.fromEntity(updated);
    }

    /** 문의 삭제 */
    @Transactional
    public void deleteInquiry(String inquiryId) {
        inquiryRepository.deleteById(inquiryId);
    }
}
