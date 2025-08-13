package com.incheonai.chatbotbackend.service;

import com.incheonai.chatbotbackend.domain.jpa.BoardCategory;
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
import org.springframework.data.jpa.domain.Specification;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class InquiryService {

    private final InquiryRepository inquiryRepository;
    private final UserRepository userRepository;

    // 검색 조건을 생성하는 private 메서드를 추가합니다.
    private Specification<Inquiry> createSpecification(String userId, BoardCategory category, String searchTerm) {
        Specification<Inquiry> spec = (root, query, cb) -> cb.conjunction();

        if (userId != null) {
            spec = spec.and((root, query, cb) -> cb.equal(root.get("userId"), userId));
        }

        if (category != null) {
            spec = spec.and((root, query, cb) -> cb.equal(root.get("category"), category));
        }

        if (searchTerm != null && !searchTerm.isBlank()) {
            Specification<Inquiry> searchSpec = (root, query, cb) ->
                    cb.or(
                            cb.like(root.get("title"), "%" + searchTerm + "%"),
                            cb.like(root.get("content"), "%" + searchTerm + "%")
                    );
            spec = spec.and(searchSpec);
        }

        return spec;
    }

    /** ✅ 전체 문의/건의 목록 조회 (GET /api/board) */
    @Transactional(readOnly = true)
    public Page<InquiryDto> getAllInquiries(String category, Pageable pageable) {
        // category 파라미터가 없거나 비어있으면 모든 목록을 반환
        if (category == null || category.isBlank()) {
            return inquiryRepository.findAll(pageable).map(InquiryDto::fromEntity);
        }

        try {
            // 문자열로 받은 category를 BoardCategory Enum 타입으로 변환
            BoardCategory boardCategory = BoardCategory.valueOf(category.toUpperCase());
            // 새로 만든 findByCategory 메서드 호출
            return inquiryRepository.findByCategory(boardCategory, pageable)
                    .map(InquiryDto::fromEntity);
        } catch (IllegalArgumentException e) {
            // "INQUIRY", "SUGGESTION"이 아닌 잘못된 문자열이 들어오면 예외 발생
            throw new InvalidDataAccessApiUsageException("유효하지 않은 카테고리 값입니다.");
        }
    }

    @Transactional(readOnly = true)
    public Page<InquiryDto> getAllInquiries(String category, String searchTerm, Pageable pageable) {
        BoardCategory boardCategory = (category != null && !category.isBlank()) ? BoardCategory.valueOf(category.toUpperCase()) : null;
        Specification<Inquiry> spec = createSpecification(null, boardCategory, searchTerm); // userId는 null
        return inquiryRepository.findAll(spec, pageable).map(InquiryDto::fromEntity);
    }

    /** 문의 등록 (POST /inquiries) */
    @Transactional
    public InquiryDto createInquiry(String userId, InquiryRequestDto request) {
        if (!userRepository.existsByUserId(userId)) {
            throw new IllegalArgumentException("존재하지 않는 사용자입니다.");
        }

        Inquiry inquiry = Inquiry.builder()
                .userId(userId)
                .title(request.title())
                .content(request.content())
                .category(request.category())
                 .status(InquiryStatus.PENDING)
                .build();

        return InquiryDto.fromEntity(inquiryRepository.save(inquiry));
    }

    @Transactional(readOnly = true)
    public Page<InquiryDto> getMyInquiries(String userId, String category, String searchTerm, Pageable pageable) {
        BoardCategory boardCategory = (category != null && !category.isBlank()) ? BoardCategory.valueOf(category.toUpperCase()) : null;
        Specification<Inquiry> spec = createSpecification(userId, boardCategory, searchTerm);
        return inquiryRepository.findAll(spec, pageable).map(InquiryDto::fromEntity);
    }

    @Transactional(readOnly = true)
    public InquiryDetailDto getInquiryDetailById(Integer inquiryId) {
        Inquiry inquiry = inquiryRepository
                .findById(inquiryId) // userId 없이 inquiryId로만 조회
                .orElseThrow(() -> new IllegalArgumentException("존재하지 않는 문의입니다."));
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
