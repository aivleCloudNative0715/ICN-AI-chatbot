package com.incheonai.chatbotbackend.repository.jpa.specification;

import com.incheonai.chatbotbackend.domain.jpa.BoardCategory;
import com.incheonai.chatbotbackend.domain.jpa.Inquiry;
import com.incheonai.chatbotbackend.domain.jpa.InquiryStatus;
import com.incheonai.chatbotbackend.domain.jpa.Urgency;
import org.springframework.data.jpa.domain.Specification;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;

public class InquirySpecification {

    public static Specification<Inquiry> hasStatus(InquiryStatus status) {
        return (root, query, criteriaBuilder) ->
                status == null ? null : criteriaBuilder.equal(root.get("status"), status);
    }

    /**
     * 중요도 목록 중 하나라도 일치하는 경우를 찾습니다 (OR 조건).
     * @param urgencies 중요도 Enum 리스트
     * @return Specification
     */
    public static Specification<Inquiry> hasUrgenciesIn(List<Urgency> urgencies) {
        return (root, query, criteriaBuilder) ->
                (urgencies == null || urgencies.isEmpty()) ?
                        null : root.get("urgency").in(urgencies);
    }

    public static Specification<Inquiry> hasCategory(BoardCategory category) {
        return (root, query, criteriaBuilder) ->
                category == null ? null : criteriaBuilder.equal(root.get("category"), category);
    }

    /**
     * 제목 또는 내용에 검색어가 포함된 경우를 찾습니다.
     * @param search 검색어
     * @return Specification
     */
    public static Specification<Inquiry> containsTextInTitleAndContent(String search) {
        if (search == null || search.isBlank()) {
            return null;
        }
        String pattern = "%" + search.toLowerCase() + "%";
        return (root, query, criteriaBuilder) ->
                criteriaBuilder.or(
                        criteriaBuilder.like(criteriaBuilder.lower(root.get("title")), pattern),
                        criteriaBuilder.like(criteriaBuilder.lower(root.get("content")), pattern)
                );
    }

    /**
     * 주어진 날짜 범위 내에 생성된 문의를 찾습니다.
     * @param start 시작일
     * @param end   종료일
     * @return Specification
     */
    public static Specification<Inquiry> isCreatedAtBetween(LocalDateTime start, LocalDateTime end) {
        return (root, query, criteriaBuilder) -> {
            if (start != null && end != null) {
                return criteriaBuilder.between(root.get("createdAt"), start, end);
            }
            if (start != null) {
                return criteriaBuilder.greaterThanOrEqualTo(root.get("createdAt"), start);
            }
            if (end != null) {
                return criteriaBuilder.lessThanOrEqualTo(root.get("createdAt"), end);
            }
            return null; // start와 end가 모두 null이면 필터링하지 않음
        };
    }
}