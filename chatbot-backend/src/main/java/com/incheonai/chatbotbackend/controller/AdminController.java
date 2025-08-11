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

import com.incheonai.chatbotbackend.dto.AdminAddRequestDto;
import com.incheonai.chatbotbackend.dto.AdminDto;
import com.incheonai.chatbotbackend.dto.InquiryAnswerRequestDto;
import com.incheonai.chatbotbackend.dto.InquiryAnswerResponseDto;
import com.incheonai.chatbotbackend.dto.InquiryDetailDto;
import com.incheonai.chatbotbackend.dto.InquiryDto;
import com.incheonai.chatbotbackend.dto.KnowledgeFileDto;
import com.incheonai.chatbotbackend.dto.StatusUpdateRequestDto;
import com.incheonai.chatbotbackend.dto.UrgencyUpdateRequestDto;
import com.incheonai.chatbotbackend.dto.ApiMessage;
import com.incheonai.chatbotbackend.service.AdminService;
import com.incheonai.chatbotbackend.service.AdminInquiryService;
import com.incheonai.chatbotbackend.service.KnowledgeFileService;

@RestController
@RequestMapping("/admin")
public class AdminController {

    private final AdminService adminService;
    private final AdminInquiryService adminInquiryService;
    private final KnowledgeFileService knowledgeFileService;

    @Autowired
    public AdminController(
            AdminService adminService,
            AdminInquiryService adminInquiryService,
            KnowledgeFileService knowledgeFileService) {
        this.adminService = adminService;
        this.adminInquiryService = adminInquiryService;
        this.knowledgeFileService = knowledgeFileService;
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
            @RequestParam(required = false) String status,
            @RequestParam(required = false) Integer urgency,
            @RequestParam(required = false) String category,
            @RequestParam(required = false) String search,
            @RequestParam int page,
            @RequestParam int size) {
        Page<InquiryDto> pageData = adminInquiryService.getInquiries(
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
        return ResponseEntity.ok(adminInquiryService.getInquiryCounts(start, end));
    }

    /** 문의 상세 조회 */
    @GetMapping("/inquiries/{inquiry_id}")
    public ResponseEntity<InquiryDetailDto> getInquiryDetail(
            @PathVariable("inquiry_id") Integer inquiryId) {
        return ResponseEntity.ok(adminInquiryService.getInquiryDetail(inquiryId.toString()));
    }

    /** 문의 긴급도 수정 */
    @PatchMapping("/inquiries/{inquiry_id}/urgency")
    public ResponseEntity<ApiMessage> updateUrgency(
            @PathVariable("inquiry_id") Integer inquiryId,
            @RequestBody UrgencyUpdateRequestDto request) {
        adminInquiryService.updateUrgency(inquiryId.toString(), request.urgency());
        return ResponseEntity.ok(new ApiMessage("긴급도가 수정되었습니다."));
    }

    /** 문의 상태 수정 */
    @PatchMapping("/inquiries/{inquiry_id}/status")
    public ResponseEntity<ApiMessage> updateStatus(
            @PathVariable("inquiry_id") Integer inquiryId,
            @RequestBody StatusUpdateRequestDto request) {
        adminInquiryService.updateStatus(inquiryId, request.status());
        return ResponseEntity.ok(new ApiMessage("문의 상태가 수정되었습니다."));
    }

    /** 답변 등록 */
    @PostMapping("/inquiries/{inquiry_id}/answers")
    public ResponseEntity<InquiryAnswerResponseDto> addAnswer(
            @PathVariable("inquiry_id") Integer inquiryId,
            @RequestBody InquiryAnswerRequestDto request) {
        InquiryAnswerResponseDto answer = adminInquiryService.addAnswer(
                inquiryId.toString(), request);
        return ResponseEntity.ok(answer);
    }

    /** 답변 수정 */
    @PatchMapping("/inquiries/{inquiry_id}/answers/{answer_id}")
    public ResponseEntity<ApiMessage> updateAnswer(
            @PathVariable("inquiry_id") Integer inquiryId,
            @PathVariable("answer_id") Integer answerId,
            @RequestBody InquiryAnswerRequestDto request) {
        adminInquiryService.updateAnswer(inquiryId, answerId, request);
        return ResponseEntity.ok(new ApiMessage("답변이 수정되었습니다."));
    }

    /** 문의 삭제 */
    @DeleteMapping("/inquiries/{inquiry_id}")
    public ResponseEntity<ApiMessage> deleteInquiry(
            @PathVariable("inquiry_id") Integer inquiryId) {
        adminInquiryService.deleteInquiry(inquiryId.toString());
        return ResponseEntity.ok(new ApiMessage("문의가 삭제되었습니다."));
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
