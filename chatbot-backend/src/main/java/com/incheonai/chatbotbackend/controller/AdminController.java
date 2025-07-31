package com.incheonai.chatbotbackend.controller;

import java.time.LocalDateTime;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.HttpHeaders;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import com.incheonai.chatbotbackend.dto.AdminCreateRequestDto;
import com.incheonai.chatbotbackend.dto.AdminDto;
import com.incheonai.chatbotbackend.dto.AdminLoginRequestDto;
import com.incheonai.chatbotbackend.dto.AdminLoginResponseDto;
import com.incheonai.chatbotbackend.dto.InquiryAnswerRequestDto;
import com.incheonai.chatbotbackend.dto.InquiryAnswerResponseDto;
import com.incheonai.chatbotbackend.dto.InquiryDetailDto;
import com.incheonai.chatbotbackend.dto.InquiryDto;
import com.incheonai.chatbotbackend.dto.KnowledgeFileDto;
import com.incheonai.chatbotbackend.dto.StatusUpdateRequestDto;
import com.incheonai.chatbotbackend.dto.UrgencyUpdateRequestDto;
import com.incheonai.chatbotbackend.service.AdminService;
import com.incheonai.chatbotbackend.service.InquiryService;
import com.incheonai.chatbotbackend.service.KnowledgeFileService;

@RestController
@RequestMapping("/admin")
public class AdminController {

    private final AdminService adminService;
    private final InquiryService inquiryService;
    private final KnowledgeFileService knowledgeFileService;

    @Autowired
    public AdminController(
            AdminService adminService,
            InquiryService inquiryService,
            KnowledgeFileService knowledgeFileService) {
        this.adminService = adminService;
        this.inquiryService = inquiryService;
        this.knowledgeFileService = knowledgeFileService;
    }

    /** 관리자 로그인 */
    @PostMapping("/login")
    public ResponseEntity<AdminLoginResponseDto> login(
            @RequestBody AdminLoginRequestDto request) {
        return ResponseEntity.ok(adminService.login(request));
    }

    /** 관리자 등록 */
    @PostMapping("/users")
    public ResponseEntity<AdminDto> createAdmin(
            @RequestBody AdminCreateRequestDto request) {
        return ResponseEntity.ok(adminService.createAdmin(request));
    }

    /** 관리자 목록 조회 */
    @GetMapping("/users")
    public ResponseEntity<Page<AdminDto>> getAdmins(
            @RequestParam int page,
            @RequestParam int size,
            @RequestParam(name = "is_active") boolean isActive) {
        Page<AdminDto> result = adminService.getAdmins(
                PageRequest.of(page, size), isActive);
        return ResponseEntity.ok(result);
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
            @RequestParam(required = false) String status,
            @RequestParam(required = false) Integer urgency,
            @RequestParam(required = false) String category,
            @RequestParam(required = false) String search,
            @RequestParam int page,
            @RequestParam int size) {
        Page<InquiryDto> pageData = inquiryService.getInquiries(
                status, urgency, category, search,
                PageRequest.of(page, size));
        return ResponseEntity.ok(pageData);
    }

    /** 문의 건수 조회 */
    @GetMapping("/inquiries/counts")
    public ResponseEntity<Map<String, Long>> getInquiryCounts(
            @RequestParam("created_at_start")
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime start,
            @RequestParam("created_at_end")
            @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime end) {
        return ResponseEntity.ok(inquiryService.getInquiryCounts(start, end));
    }

    /** 문의 상세 조회 */
    @GetMapping("/inquiries/{inquiry_id}")
    public ResponseEntity<InquiryDetailDto> getInquiryDetail(
            @PathVariable("inquiry_id") Integer inquiryId) {
        return ResponseEntity.ok(inquiryService.getInquiryDetail(inquiryId.toString()));
    }

    /** 문의 긴급도 수정 */
    @PatchMapping("/inquiries/{inquiry_id}/urgency")
    public ResponseEntity<Void> updateUrgency(
            @PathVariable("inquiry_id") Integer inquiryId,
            @RequestBody UrgencyUpdateRequestDto request) {
        inquiryService.updateUrgency(inquiryId.toString(), request.getUrgency());
        return ResponseEntity.noContent().build();
    }

    /** 문의 상태 수정 */
    @PatchMapping("/inquiries/{inquiry_id}/status")
    public ResponseEntity<Void> updateStatus(
            @PathVariable("inquiry_id") Integer inquiryId,
            @RequestBody StatusUpdateRequestDto request) {
        inquiryService.updateStatus(inquiryId.toString(), request.getStatus());
        return ResponseEntity.noContent().build();
    }

    /** 답변 등록 */
    @PostMapping("/inquiries/{inquiry_id}/answers")
    public ResponseEntity<InquiryAnswerResponseDto> addAnswer(
            @PathVariable("inquiry_id") Integer inquiryId,
            @RequestBody InquiryAnswerRequestDto request) {
        InquiryAnswerResponseDto answer = inquiryService.addAnswer(
                inquiryId.toString(), request);
        return ResponseEntity.ok(answer);
    }

    /** 답변 수정 */
    @PatchMapping("/inquiries/{inquiry_id}/answers/{answer_id}")
    public ResponseEntity<Void> updateAnswer(
            @PathVariable("inquiry_id") Integer inquiryId,
            @PathVariable("answer_id") Integer answerId,
            @RequestBody InquiryAnswerRequestDto request) {
        inquiryService.updateAnswer(
                inquiryId.toString(), answerId.toString(), request);
        return ResponseEntity.noContent().build();
    }

    /** 문의 삭제 */
    @DeleteMapping("/inquiries/{inquiry_id}")
    public ResponseEntity<Void> deleteInquiry(
            @PathVariable("inquiry_id") Integer inquiryId) {
        inquiryService.deleteInquiry(inquiryId.toString());
        return ResponseEntity.noContent().build();
    }

    /** 파일 목록 조회 */
    @GetMapping("/knowledge-files")
    public ResponseEntity<Page<KnowledgeFileDto>> getKnowledgeFiles(
            @RequestParam int page,
            @RequestParam int size) {
        Page<KnowledgeFileDto> files = knowledgeFileService.getFiles(
                PageRequest.of(page, size));
        return ResponseEntity.ok(files);
    }

    /** 파일 업로드 */
    @PostMapping("/knowledge-files")
    public ResponseEntity<KnowledgeFileDto> uploadFile(
            @RequestParam("file") MultipartFile file,
            @RequestParam("description") String description) {
        KnowledgeFileDto dto = knowledgeFileService.uploadFile(file, description);
        return ResponseEntity.ok(dto);
    }

    /** 파일 삭제 */
    @DeleteMapping("/knowledge-files/{file_id}")
    public ResponseEntity<Void> deleteFile(
            @PathVariable("file_id") Integer fileId) {
        knowledgeFileService.deleteFile(fileId.toString());
        return ResponseEntity.noContent().build();
    }

    /** 파일 다운로드 */
    @GetMapping("/knowledge-files/{file_id}/download")
    public ResponseEntity<Resource> downloadFile(
            @PathVariable("file_id") Integer fileId) {
        Resource resource = knowledgeFileService.loadFileAsResource(fileId.toString());
        String header = "attachment; filename=\"" + resource.getFilename() + "\"";
        return ResponseEntity.ok()
                .header(HttpHeaders.CONTENT_DISPOSITION, header)
                .body(resource);
    }
}
