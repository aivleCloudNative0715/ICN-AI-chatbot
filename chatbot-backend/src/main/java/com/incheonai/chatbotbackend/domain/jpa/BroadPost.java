package com.incheonai.chatbotbackend.domain.jpa;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import java.time.LocalDateTime;

@Entity
@Table(name = "broad_posts")
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
public class BroadPost {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long postId;

    @Column(nullable = false)
    private String authorId;

    @Column(nullable = false)
    private String title;

    @Column(nullable = false, columnDefinition = "TEXT")
    private String content;

    @Column(nullable = false)
    private String category;

    @CreationTimestamp
    private LocalDateTime createdAt;

    @UpdateTimestamp
    private LocalDateTime updatedAt;

    @Column(nullable = false)
    private Boolean isDeleted = false;

    @Builder
    public BroadPost(String authorId, String title, String content, String category) {
        this.authorId = authorId;
        this.title    = title;
        this.content  = content;
        this.category = category;
    }

    public void update(String title, String content, String category) {
        this.title    = title;
        this.content  = content;
        this.category = category;
    }

    public void delete() {
        this.isDeleted = true;
    }
}
