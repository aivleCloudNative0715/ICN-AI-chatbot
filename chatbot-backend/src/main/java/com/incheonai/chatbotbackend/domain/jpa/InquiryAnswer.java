package com.incheonai.chatbotbackend.domain.jpa;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import java.time.LocalDateTime;

@Entity
@Table(name = "inquiry_answers")
@Getter
@NoArgsConstructor
public class InquiryAnswer {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer answerId;

    @Column(nullable = false)
    private Integer inquiryId;

    @Column(nullable = false)
    private String adminId;

    @Column(nullable = false, columnDefinition = "TEXT")
    private String content;

    @CreationTimestamp
    private LocalDateTime createdAt;

    @UpdateTimestamp
    private LocalDateTime updatedAt;

    @Builder
    public InquiryAnswer(Integer inquiryId, String adminId, String content) {
        this.inquiryId = inquiryId;
        this.adminId   = adminId;
        this.content   = content;
    }

    /** 답변 내용 수정용 Setter */
    public void setContent(String content) {
        this.content = content;
    }
}
