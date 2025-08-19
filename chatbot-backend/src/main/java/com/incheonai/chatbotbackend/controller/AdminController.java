package com.incheonai.chatbotbackend.controller;

import java.time.LocalDate;
import java.util.List;

import com.incheonai.chatbotbackend.domain.jpa.BoardCategory;
import com.incheonai.chatbotbackend.domain.jpa.InquiryStatus;
import com.incheonai.chatbotbackend.domain.jpa.Urgency;
import com.incheonai.chatbotbackend.dto.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.data.web.PageableDefault;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import com.incheonai.chatbotbackend.service.AdminService;
import com.incheonai.chatbotbackend.service.AdminInquiryService;

@RestController
@RequestMapping("api/admin")
public class AdminController {

    private final AdminService adminService;
    private final AdminInquiryService adminInquiryService;

    @Autowired
    public AdminController(
            AdminService adminService,
            AdminInquiryService adminInquiryService) {
        this.adminService = adminService;
        this.adminInquiryService = adminInquiryService;
    }

    /** 관리자 추가 */
    @PostMapping("/users")
    public ResponseEntity<AdminDto> addAdmin(
            @RequestBody AdminAddRequestDto request) {
        return ResponseEntity.ok(
                adminService.addAdmin(request)
        );
    }

    /** 관리자 목록 조회 */
    @GetMapping("/users")
    public ResponseEntity<Page<AdminDto>> getAdmins(
            @RequestParam int page,
            @RequestParam int size,
            @RequestParam(name = "is_active") boolean isActive) {
        return ResponseEntity.ok(
                adminService.getAdmins(PageRequest.of(page, size), isActive)
        );
    }

    /** 관리자 삭제 */
    @DeleteMapping("/users/{admin_id}")
    public ResponseEntity<Void> deleteAdmin(
            @PathVariable("admin_id") String adminId) {
        adminService.deleteAdmin(adminId);
        return ResponseEntity.noContent().build();
    }

    /** 문의 목록 조회 */
    @GetMapping("/inquiries")
    public ResponseEntity<Page<InquiryDto>> getAllInquiries(
            @RequestParam(required = false) InquiryStatus status,
            @RequestParam(required = false) List<Urgency> urgency,
            @RequestParam(required = false) BoardCategory category,
            @RequestParam(required = false) String search,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate start,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate end,
            @PageableDefault(sort = "inquiryId", direction = Sort.Direction.DESC) Pageable pageable) {
        Page<InquiryDto> pageData = adminInquiryService.getInquiries(
                status, urgency, category, search, start, end, pageable);
        return ResponseEntity.ok(pageData);
    }

    /** 문의 건수 조회 */
    @GetMapping("/inquiries/counts")
    public ResponseEntity<InquiryCountsDto> getInquiryCounts() {
        return ResponseEntity.ok(adminInquiryService.getInquiryCounts());
    }

    /** 문의 상세 조회 */
    @GetMapping("/inquiries/{inquiry_id}")
    public ResponseEntity<InquiryDetailDto> getInquiryDetail(
            @PathVariable("inquiry_id") Integer inquiryId) {
        return ResponseEntity.ok(adminInquiryService.getInquiryDetail(inquiryId));
    }

    /** 문의 긴급도 수정 */
    @PatchMapping("/inquiries/{inquiry_id}/urgency")
    public ResponseEntity<ApiMessage> updateUrgency(
            @PathVariable("inquiry_id") Integer inquiryId,
            @RequestBody UrgencyUpdateRequestDto request) { // DTO 타입 변경
        adminInquiryService.updateUrgency(inquiryId, request.urgency());
        return ResponseEntity.ok(new ApiMessage("긴급도가 수정되었습니다."));
    }

    /** 답변 등록/수정 (API 통합) */
    @PostMapping("/inquiries/{inquiry_id}/answer")
    public ResponseEntity<InquiryDetailDto> processAnswer(
            @PathVariable("inquiry_id") Integer inquiryId,
            @RequestBody InquiryAnswerRequestDto request) {
        InquiryDetailDto updatedInquiry = adminInquiryService.processAnswer(inquiryId, request);
        return ResponseEntity.ok(updatedInquiry);
    }
}
