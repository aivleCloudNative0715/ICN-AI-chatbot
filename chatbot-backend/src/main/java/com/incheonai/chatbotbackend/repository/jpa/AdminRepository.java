package com.incheonai.chatbotbackend.repository.jpa;

import com.incheonai.chatbotbackend.domain.jpa.Admin;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface AdminRepository extends JpaRepository<Admin, Integer> {
    Optional<Admin> findByAdminId(String adminId);
    // soft-delete 되지 않은(=deletedAt IS NULL) 활성 관리자 페이징 조회
    Page<Admin> findAllByDeletedAtIsNull(Pageable pageable);
    // soft-delete된(=deletedAt IS NOT NULL) 관리자 페이징 조회
    Page<Admin> findAllByDeletedAtIsNotNull(Pageable pageable);
}
