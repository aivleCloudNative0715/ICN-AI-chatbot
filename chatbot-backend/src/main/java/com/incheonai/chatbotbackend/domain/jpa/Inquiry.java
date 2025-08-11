package com.incheonai.chatbotbackend.domain.jpa;


import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.SQLDelete;
import org.hibernate.annotations.UpdateTimestamp;
import org.hibernate.annotations.Where;

import java.time.LocalDateTime;

@Getter
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

    @Column(nullable = false)
    private String category;

    @Column(columnDefinition = "TEXT")
    private String answer;

    @Column(name = "admin_id")
    private String adminId;

    @Column
    private Integer urgency;

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
    public Inquiry(String userId, String title, String content, String category,
                   Integer urgency, InquiryStatus status) {

        this.userId = userId;
        this.title = title;
        this.content = content;
        this.category = category;
        this.urgency = urgency;
        this.status = (status == null) ? InquiryStatus.PENDING : status;
    }

    // Setters for mutable fields
    public void setTitle(String title) {
        this.title = title;
    }

    public void setContent(String content) {
        this.content = content;
    }

    public void setCategory(String category) {
        this.category = category;
    }

    public void setUrgency(Integer urgency) {
        this.urgency = urgency;
    }

    public void setStatus(InquiryStatus status) {
        this.status = status;
    }

    public void setAnswer(String answer) {
        this.answer = answer;
    }

    public void setAdminId(String adminId) {
        this.adminId = adminId;
    }
}
