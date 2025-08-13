package com.incheonai.chatbotbackend.repository.jpa.specification;

import com.incheonai.chatbotbackend.domain.jpa.BoardCategory;
import com.incheonai.chatbotbackend.domain.jpa.Inquiry;
import com.incheonai.chatbotbackend.domain.jpa.InquiryStatus;
import com.incheonai.chatbotbackend.domain.jpa.Urgency;
import org.springframework.data.jpa.domain.Specification;

public class InquirySpecification {

    public static Specification<Inquiry> hasStatus(InquiryStatus status) {
        return (root, query, criteriaBuilder) ->
                status == null ? null : criteriaBuilder.equal(root.get("status"), status);
    }

    public static Specification<Inquiry> hasUrgency(Urgency urgency) {
        return (root, query, criteriaBuilder) ->
                urgency == null ? null : criteriaBuilder.equal(root.get("urgency"), urgency);
    }

    public static Specification<Inquiry> hasCategory(BoardCategory category) {
        return (root, query, criteriaBuilder) ->
                category == null ? null : criteriaBuilder.equal(root.get("category"), category);
    }

    public static Specification<Inquiry> containsText(String search) {
        if (search == null || search.isBlank()) {
            return null;
        }
        String pattern = "%" + search.toLowerCase() + "%";
        return (root, query, criteriaBuilder) ->
                criteriaBuilder.or(
                        criteriaBuilder.like(criteriaBuilder.lower(root.get("title")), pattern),
                        criteriaBuilder.like(criteriaBuilder.lower(root.get("content")), pattern),
                        criteriaBuilder.like(criteriaBuilder.lower(root.get("userId")), pattern)
                );
    }
}