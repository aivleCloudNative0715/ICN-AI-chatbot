package com.incheonai.chatbotbackend.domain.jpa;


import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.SQLDelete;
import org.hibernate.annotations.UpdateTimestamp;
import org.hibernate.annotations.Where;

import java.time.LocalDateTime;

@Getter
@Setter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Entity
@Table(name = "inquiries")
@SQLDelete(sql = "UPDATE inquiries SET deleted_at = CURRENT_TIMESTAMP WHERE inquiry_id = ?")
@Where(clause = "deleted_at IS NULL")
public class Inquiry {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "inquiry_id")
    private Integer inquiryId;

    @Column(name = "user_id", nullable = false)
    private String userId;

    @Column(nullable = false)
    private String title;

    @Column(columnDefinition = "TEXT", nullable = false)
    private String content;

    @Enumerated(EnumType.STRING) // DB에 Enum 이름을 문자열로 저장
    @Column(nullable = false)
    private BoardCategory category;

    @Column(columnDefinition = "TEXT")
    private String answer;

    @Column(name = "admin_id")
    private String adminId;

    @Enumerated(EnumType.STRING)
    @Column
    private Urgency urgency;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private InquiryStatus status;

    @CreationTimestamp
    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;

    @UpdateTimestamp
    @Column(name = "updated_at")
    private LocalDateTime updatedAt;

    @Column(name = "deleted_at")
    private LocalDateTime deletedAt;

    @Builder
    public Inquiry(String userId, String title, String content, BoardCategory category, InquiryStatus status) {
        this.userId = userId;
        this.title = title;
        this.content = content;
        this.category = category;
        this.status = (status == null) ? InquiryStatus.PENDING : status;
        this.urgency = Urgency.MEDIUM; // 사용자가 등록 시 '보통'으로 기본값 설정
    }
}
