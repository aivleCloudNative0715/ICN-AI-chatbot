package com.incheonai.chatbotbackend.service;

import com.incheonai.chatbotbackend.domain.jpa.Admin;
import com.incheonai.chatbotbackend.dto.AdminAddRequestDto;
import com.incheonai.chatbotbackend.dto.AdminDto;
import com.incheonai.chatbotbackend.repository.jpa.AdminRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class AdminService {

    private final AdminRepository adminRepository;
    private final PasswordEncoder passwordEncoder;

    /**
     * 관리자 계정 추가
     * @param request 관리자 추가 요청 DTO
     * @return 생성된 관리자 정보를 담은 DTO
     */
    @Transactional
    public AdminDto addAdmin(AdminAddRequestDto request) {
        // 1. 요청 DTO로부터 Admin 엔티티 생성 (비밀번호는 암호화)
        Admin admin = Admin.builder()
                .adminId(request.adminId())
                .adminName(request.adminName())
                .password(passwordEncoder.encode(request.password()))
                .role(request.role())
                .build();
        // 2. 저장소에 저장
        Admin saved = adminRepository.save(admin);
        // 3. 저장된 엔티티를 DTO로 변환하여 반환
        return AdminDto.fromEntity(saved);
    }

    /**
     * 관리자 목록 조회
     * @param pageable 페이징 정보 (페이지 번호, 크기, 정렬 등)
     * @param isActive 활성화 여부 필터 (true: 활성 관리자, false: 삭제된 관리자)
     * @return 페이지 단위의 AdminDto 리스트
     */
    @Transactional(readOnly = true)
    public Page<AdminDto> getAdmins(Pageable pageable, boolean isActive) {
        // 활성 상태에 따라 서로 다른 JPA 쿼리 메서드를 호출
        if (isActive) {
            // deletedAt이 NULL인 엔티티(soft-delete되지 않은 활성 관리자)만 조회
            return adminRepository
                    .findAllByDeletedAtIsNull(pageable)
                    .map(AdminDto::fromEntity);
        } else {
            // deletedAt이 NOT NULL인 엔티티(soft-delete된 관리자)만 조회
            return adminRepository
                    .findAllByDeletedAtIsNotNull(pageable)
                    .map(AdminDto::fromEntity);
        }
    }

    /**
     * 관리자 계정 삭제 (soft delete)
     * @param adminId 삭제할 관리자 로그인 ID
     */
    @Transactional
    public void deleteAdmin(String adminId) {
        // 1. 로그인 ID로 관리자 조회, 없으면 예외 발생
        Admin admin = adminRepository.findByAdminId(adminId)
                .orElseThrow(() -> new IllegalArgumentException("관리자가 없습니다."));
        // 2. soft-delete 전략에 따라 delete() 호출 시 deletedAt 컬럼이 자동으로 채워짐
        adminRepository.delete(admin);
        // 별도 save() 호출 불필요 (Hibernate가 삭제 상태 변경을 관리)
    }
}
